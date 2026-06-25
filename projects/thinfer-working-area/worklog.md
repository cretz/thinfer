# Worklog

Forward-looking only -- git history is the changelog, the code is the record.
Past work appears here ONLY as a one-line lesson or a do-not-retry. Engine-wide
design + kernel/runtime state: `plan-details.md`. Per-model porting: `wan-plan.md`
(Wan2.2-TI2V-5B line; backbone / RoPE3D / umT5 / VAE / GGUF lore reusable,
SkyReels-DF obsolete). Z-Image: `zimage-plan.md`. Ideogram-4: `ideogram-plan.md`.
Qwen-Image-Edit-Rapid (NEW, active): `qwen-image-plan.md`. Scratch is ephemeral
and clearable -- nothing here depends on a scratch file.

## Status

- **FastWan2.2-TI2V-5B-FullAttn** -- shipped baseline, parity GREEN. bf16 acts
  (DiT + umT5). DO NOT DISTURB.
- **LongLive-2.0-5B** (AR/causal long + multi-shot video, same Wan base) --
  shipped: AR path, multi-shot scene-cuts, self-attn-qkv i8 all landed, e2e
  health-GREEN, parity GREEN (two-tier bands). Remaining = AR perf + a multi-shot
  pyref byte-parity (see the LongLive section).
- **Face-swap** (`thinfer generate face-swap`) -- shipped incl. the 4K / B-frame /
  streaming / audio correctness pass. NEXT = quality (XSeg occlusion + GFPGAN
  enhancer + use HyperSwap's own mask output); see `faceswap-plan.md`.
- **Ideogram-4** (`ideogram4-q8` image model) -- shipped: Q8_0 encoder+DiT, turbotime
  LoRA folded to Q8_0, FLUX.2 KL VAE; i8 DP4A on. 512x512/4-step ~79s. DO NOT DISTURB.
  Do-not-retry: a Q4_K DiT default was tried + dropped (per-request whole-DiT fold
  re-quantizes every request, so Q4_K is ~2x slower than Q8_0 AND quality broke for
  an unrooted GPU-path reason; the on-disk size win does not pay). git log has the why.
- **i8 DP4A matmul is ON by default** (opt out `--no-i8-matmul` = bf16 reference
  path); see the i8 lesson below.

## NEXT (active): Qwen-Image-Edit-Rapid-AIO port -- FOUNDATION landed

See `qwen-image-plan.md` (full architecture audit + build order + CONFIRM list).
20B dual-stream MMDiT edit model, 4-step CFG-free, GGUF-only (Phil2Sat recipe).
Template = ideogram4. Reuse: wan/rope3d (complex interleaved RoPE, NOT ideogram
half-rot), wan/vae (Wan-family 3D-causal KL), matmul/sdpa/i8/MemArbiter as-is.
New: dual-stream block, Qwen2.5-VL encoder block (qkv bias + GQA, no qk-norm),
vision tower, edit conditioning (VAE-latent concat + vision-pad tokens).

LANDED this pass (verified):
- `qwen_image/{mod.rs,manifest.rs}` scaffold + config consts; wired in lib.rs;
  builds green. Manifest = DiT Q8_0(canary)/Q4_K_M(default) + Qwen2.5-VL-7B-
  abliterated Q8_0 + mmproj-f16 + calcuis pig qwen_image VAE + tokenizer.
- DiT architecture CONFIRMED against the Q8_0 GGUF metadata: arch=qwen_image,
  60 layers, dims per config.rs, NO guidance/layer3d, tensor names 1:1 with
  diffusers (no rename map). See plan CONFIRM list for the verified specifics.
- third-party clones for reading (top-level `/c/work/personal/thinfer/
  third-party/`, OUTSIDE repo): `diffusers` (qwenimage transformer/pipelines/
  vae, sparse) + `transformers` (qwen2_5_vl, sparse). NOT committed.
- Weights downloading to HF cache via `scratch/dl-qwen.sh` (log
  `scratch/dl-qwen.log`). Order: Q8 DiT + Q4 DiT (BOTH done) -> encoder Q8 ->
  mmproj -> VAE -> tokenizer. CHECK LIVENESS the RIGHT way: `hf` uses the XET
  backend now, so the blob `.incomplete` and `du` of the cache DO NOT grow during
  transfer (xet streams chunks, assembles the final blob only at the end). The
  authoritative liveness signal is the newest `~/.cache/huggingface/xet/logs/
  xet_*.log`: a climbing "observed bytes sent so far" = healthy. `du`/`.incomplete`
  flatness is NOT a stall under xet (cumulative size lies even harder here; see
  [[feedback_verify_bg_progress]]). A REAL stall = xet log idle + blob mtime frozen
  for many minutes. LESSON THIS SESSION (RESOLVED): the `hf` XET backend hung
  repeatedly (3x) ~2.5GB into the 8GB encoder file (DiT files pulled fine, so not
  the network). FIX THAT WORKED: kill the hung procs (`taskkill //F //PID
  <winpid>` for both the `/hf` launcher + its python child) and re-run with XET
  DISABLED: `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 bash
  scratch/dl-qwen.sh`. The classic HTTPS path writes directly to the blob
  `.incomplete` (so `.incomplete` GROWTH is the real liveness signal again, NOT
  flat-under-xet), resumed reliably at ~17MiB/s. If a future hf download hangs,
  go straight to DISABLE_XET. NOT committed (user tests first).
- GGUF metadata/tensor-name inspect tool: `scratch/gguf_dump.py`
  (`uv run --with gguf python scratch/gguf_dump.py <file.gguf>`) -- dumps KV
  header + tensor names/shapes/types. Use it for every CONFIRM item. NB GGUF KV
  header carries only arch/quant/file_type; ALL hyperparams come from tensor
  names+shapes.

STATUS (Qwen-Image t2i): RUNNING + E2E TESTED. Steps 2-5 DONE, all parity GREEN
individually (vae slope .999/rel .99%, encoder .998/4.5%, dit temb .999 block0
~2% vel .999/1.8%, e2e health sane image 78s @64x64/4step). LESSON: run the
qwen_image conformance tests ONE AT A TIME (`--release <name> -- --test-threads=1`),
NOT all together -- multiple WgpuBackend in one test binary contend on VRAM and a
later test's engine forward fails (e2e-then-encoder). Each passes in isolation.
The heavy pyrefs (encoder 7B ~15GB, dit/e2e) cache their dumps; pre-seed
`target/tmp/qwen_*` to skip re-running. NEXT = step 6 edit path (BLOCKED, see below).

NEXT (build order, smallest closed loop first):
1. Confirm encoder/mmproj/VAE specifics via `gguf_dump.py` (CONFIRM list): for
   the VAE, the tensor-key form (calcuis "pig" packs may rename vs diffusers)
   + the latents_mean/std 16-vectors + base_dim; for the encoder, layer count /
   GQA kv-heads / qkv-bias / rope_theta / tensor keys; for mmproj, vision
   depth/hidden/heads/patch/merge.
2. VAE decode (frame=1) + vae_parity vs pyref-on-same-GGUF. CONFIRMED from the
   pig VAE GGUF (243M, arch=`pig`, 194 tensors, KV header = arch/quant/file_type
   only): base_dim/decoder_base_dim=96 (`decoder.conv1` out 384 = 96*4,
   `decoder.head.0.gamma`=96), z_dim=16. NO latents_mean/std in the GGUF -> the
   diffusers defaults baked in vae.rs are authoritative (keep them). RENAME MAP
   DECODED + validated (pig = ORIGINAL Wan flat naming; `wan/vae.rs` loader wants
   DIFFUSERS naming -> map diffusers-key the loader builds -> pig-key):
     post_quant_conv <- conv2 ; decoder.conv_in <- decoder.conv1 ;
     decoder.norm_out <- decoder.head.0 ; decoder.conv_out <- decoder.head.2 ;
     decoder.mid_block.resnets.{0,1} <- decoder.middle.{0,2} ;
     decoder.mid_block.attentions.0 <- decoder.middle.1 (norm/to_qkv/proj match) ;
     decoder.up_blocks.{i}.resnets.{j} <- decoder.upsamples.{i*4+j} ;
     decoder.up_blocks.{i}.upsamplers.0 <- decoder.upsamples.{i*4+3} (i=0,1,2) ;
     resnet subkeys norm1/conv1/norm2/conv2/conv_shortcut <-
       residual.0/residual.2/residual.3/residual.6/shortcut.
   Flat layout (15) = 4 up_blocks x 3 resnets + 3 upsamplers @ flat 3/7/11.
   Topology matches the loader's `has_shortcut`=(rin!=cout) EXACTLY: only
   upsamples.4 (blk1 resnet0, 192->384) has a shortcut, confirmed in the GGUF.
   TRANSFORM LANDED: `qwen_image/vae.rs::pig_to_diffusers_vae_key` (the heart of
   the shim) + tests over EVERY dumped pig key form (incl. the upsamples.4
   shortcut, spatial-only upsamples.11, last-block 12..14). LANDED + parity GREEN
   (slope 0.999, rel 0.99% @ 64x64; tol slope 1+-0.05 / rel 8%):
   - `qwen_image/vae.rs::vae_source<S>` wraps the pig `GgufSource` in
     `thinfer_core::format::union::RenamedSource::with_passthrough` (NO new shim
     needed -- RenamedSource already existed; build the rename map by walking the
     catalog + `pig_to_diffusers_vae_key`, keeping only changed keys).
   - decode reuses `wan/vae.rs` verbatim: `register_decoder(&residency,
     &VaeDecoderWeights::new(&qwen_image_vae()))` + `WanVaePipelines::compile` +
     `WanVaeDecoder{..}.decode(.., f=1, h_in, w_in)`. wan/vae.rs byte-untouched.
   - test `tests/qwen_image/{main,parity_util,vae_parity}.rs` (feature
     `qwen-image-e2e`); pyref `python/.../qwen_image/gen_vae_ref.py` loads the SAME
     pig GGUF into diffusers `AutoencoderKLQwenImage` (decoder + post_quant_conv,
     strict=False), feeds an identical normalized latent, denorms `z*std+mean`.
   - CORE CHANGE (was blocking): the pig GGUF ships conv weights as **F16**, which
     the host `weight::Decoder` rejected ("fp16 unsupported, Z-Image M1"). Added
     F16 support: `Decoder` F16 arm (shared 2-byte carry via `feed_pairs`),
     `residency::gpu_encoding` F16->Bf16, a `narrow_f16_to_bf16`, and the F16 upload
     arm in `read_for_gpu`. ALSO fixed a latent GGUF-padding bug: the F32/F16 narrow
     arms must slice `elements*size`, not `on_disk_bytes` (GGUF pads each tensor to
     32B; e.g. `conv_out.bias` [3] -> 32B on disk). Behavior-identical for unpadded
     safetensors; core lib tests + ideogram/zimage paths unaffected.
   CAVEAT (unchanged): the pig VAE GGUF is DECODE-ONLY (no `encoder.*`; only
   `conv1`/`conv2` + `decoder.*`), so the step-6 edit path (VAE-encode of the ref
   image) CANNOT use it -- needs a separate encoder-bearing VAE source (CONFIRM
   where Phr00t's edit pipeline gets encode weights).
3. Qwen2.5-VL LM encoder (text-only) + encoder_parity. LANDED + parity GREEN
   (slope 0.998, rel 4.5%; tol slope 1+-0.03 / rel 6% -- rel is Qwen2.5 massive
   activations, range +-140, tracked by bf16 to rounding, same as ideogram ~7%).
   `qwen_image/text_encoder.rs`: NEW Qwen2.5 block CLONED from
   `z_image::text_encoder::Qwen3Block` with the two deltas: (a) QKV BIAS added
   (matmul via `Block::dispatch_matmul_site` then `bcast_add::<BcastAddF32>` per
   q/k/v, mirroring `embedders::linear_bias`); (b) q/k-norm DELETED (Qwen2.5 has
   none). GQA 28Q/4KV, head_dim 128, SwiGLU, 1-axis half-rot RoPE (text MRoPE
   collapses since t=h=w=pos; GGUF sections [16,24,24]). Runs ALL 28 layers +
   final `output_norm` -> `hidden_states[-1]` (Qwen-Image takes [-1] post-norm,
   NOT z_image's [-2]). Reuses z_image `register_one` + `embed_lookup_hidden`
   (pub(crate), name-agnostic) via `qwen2vl_gguf_renames()` (native blk.N ->
   model.layers.N, incl. bias keys; wrap source in RenamedSource). bf16 acts, all
   matmul sites Q8_0. pyref `gen_encoder_ref.py` builds HF `Qwen2_5_VLTextModel`
   (transformers 5.x; explicit dims + rope_scaling mrope_section [16,24,24]),
   loads the SAME GGUF (dequant Q8_0->bf16), position_ids all-equal across 3
   axes. Test `tests/qwen_image/encoder_parity.rs`. NB pyref is 7B bf16 ~15GB,
   ~2min; run as own process.
4. MMDiT block + full DiT + dit_parity. ROPE LANDED (block prerequisite, fully
   unblocked + tested, no GGUF dep): `qwen_image/rope.rs::QwenImageRope` ->
   `vid_freqs(frame,h,w)` + `txt_freqs(h,w,txt_len)`, `[seq, head_dim=128]`
   interleaved (re,im) = the shared rope-kernel layout. Reuses `RopeEmbedder`
   (axes [16,56,56], theta 1e4); `scale_rope` centering done via a NEW shared
   `RopeEmbedder::lookup_signed` (negative pos = complex conjugate; positive path
   bit-identical to `lookup`, so z_image/wan untouched). Ported verbatim from
   `QwenEmbedRope._compute_video_freqs`. Tests green. REMAINING for the block:
   img_in/txt_in/txt_norm + time_text_embed + 60x dual-stream (joint attn, qk-norm,
   img/txt modulation) + norm_out/proj_out, on the GPU matmul/sdpa/i8 stack; can't
   dit_parity until encoder+VAE GGUFs land (downloading).
   PACKING LANDED (pipeline prerequisite, host-side + round-trip tested):
   `qwen_image/packing.rs::{pack_latents,unpack_latents}` latent[16,H,W] <->
   tokens[(H/2)(W/2),64], CHANNEL-major `c*4+ph*2+pw` (NOT ideogram's patch-major
   `[ph,pw,c]`). Verbatim from `_pack_latents`/`_unpack_latents`.
   DIT LANDED + single-block parity GREEN (temb slope 1.000/rel 0.72%, block0_hs
   slope 1.0005/rel 2.30%, block0_eh slope 0.998/rel 0.91%, velocity slope 0.999/
   rel 1.77%; tol temb 3% / block0 6% / vel slope 0.06 rel 15%). Files:
   `qwen_image/{dit.rs,loader.rs}`. dit.rs = `QwenImageDitPipelines` (BlockPipelines
   + a gelu pipeline, mirrors WanDitPipelines) + local op wrappers copied from
   wan/dit_block (op_layernorm/op_modulate/op_gate_residual/op_bias_add) + the
   dual-stream `block_forward` + the `QwenImageDit` driver (img_in, txt_norm+txt_in,
   timestep sinusoid->linear_1->SiLU->linear_2, SiLU(temb) shared, 1-axis... no:
   complex interleaved RoPE via `op_rope` [made pub(crate)], block loop, norm_out
   AdaLayerNormContinuous [scale,shift = chunk(emb,2), scale FIRST] + proj_out).
   Block: per-stream Linear(SiLU(temb))->[1,6*dim] sliced [shift,scale,gate]_{msa,
   mlp}; LayerNorm(no-affine)->modulate; joint attn = separate to_q/k/v + add_q/k/v
   (BIAS) -> per-head QK-RMSNorm -> op_rope (img=vid_freqs, txt=txt_freqs) -> concat
   [txt++img] (copy_buffer_to_buffer) -> ONE op_sdpa (has_mask=0) -> split -> to_out/
   to_add_out -> gate1 residual; norm2->modulate->GELU MLP (dim->4dim->dim)->gate2.
   Block matmuls Q8_0 (qkv site); embedders F16->bf16 (adaln site); bf16 acts.
   CORE/LOADER changes (small, general): `op_rope` -> pub(crate);
   `common/loader.rs::register_linear_transcode` accepts F16 (Linear2D->bf16, the
   img_in/proj_out/time_embed dtype); `weight::Decoder` F16 already added in step 2.
   pyref `gen_dit_ref.py` = 1-LAYER `QwenImageTransformer2DModel` (full 60-layer
   bf16 is ~41GB, won't fit 64GB; 1 block validates all kernels, depth is just
   repetition), loads block0+embedders from the SAME GGUF, hooks temb + block0,
   dumps seeded img_tokens/txt_embeds + velocity. Driver runs N blocks (1 parity,
   60 e2e). DO-NOT-RETRY: full-DiT bf16 pyref OOMs; e2e (step 5) full-DiT reference
   must stream or rely on component parity + visual eyeball.
5. t2i pipeline + e2e -- LANDED + e2e health GREEN (full pipeline runs in 78s @
   64x64/4steps: 46 tokens -> encode -> 60-block DiT streamed x4 -> VAE -> RGB
   min -1.0/max 1.0/mean -0.10/std 0.68, finite + in-range + non-degenerate).
   `qwen_image/pipeline.rs::QwenImagePipeline<S>`: load() compiles encoder
   (BlockPipelines bf16/Q8_0) + DiT (QwenImageDitPipelines) + VAE (WanVaePipelines)
   over ONE union residency; generate_rgb does encode -> evict -> FlowMatchEuler
   denoise (CFG-free) -> evict -> VAE decode -> RGB, phase eviction like ideogram.
   SAMPLER `build_steps`: sigmas=linspace(1,1/N,N)++0; mu=calculate_shift(img_seq,
   256,4096,0.5,1.15); shifted=e^mu/(e^mu+(1/sigma-1)); DiT timestep=shifted[i]
   (=scheduler_t/1000), z += (shifted[i+1]-shifted[i])*velocity. DROP_IDX=34 (t2i
   template preamble dropped from encoder output). Latents host-side between steps
   (re-upload each step; perf step 7). e2e test `tests/qwen_image/e2e.rs` + fast
   tokenizer-only pyref `tokenize_prompt.py`; union = GgufSource(dit 1:1) +
   RenamedSource(enc, qwen2vl) + vae_source(vae). NB std 0.68 high @ 64px/4step
   (low-res may not fully denoise); for a real visual check run larger dims.
   NEXT: edit path (vision tower + VAE encode + latent concat) = step 6.
6. EDIT PATH -- LANDED + e2e HEALTH GREEN (2026-06-25). The full image->image edit
   model runs end to end (min -1 / max 1 / mean -0.35 / std 0.63 @ 64x64/4step, PNG
   plausible). ALL components parity-green: VAE decode (.9989/.99%) + encode
   (1.003/.23%), vision tower (8x8 2.80% / 16x16 4.39%), encoder MRoPE+scatter
   (.9989/4.97%), DiT t2i (prior), DiT edit-concat exercised by edit_e2e. Pipeline:
   `QwenImagePipeline::generate_edit_rgb` (vision.forward -> drop-64 forward_edit ->
   VAE-encode ref + normalize_ref_latent + pack -> CFG-free FlowMatchEuler over
   [noise++ref] via dit.forward_multi -> decode). New: qwen_image/vision.rs,
   text_encoder::{forward_edit,EditEncoderPipelines,build_mrope_freqs,
   build_edit_position_ids}, core ops::RopeF32Mrope, dit::forward_multi,
   rope::{vid,txt}_freqs_multi, vae::normalize_ref_latent. Tests: vision_parity,
   encoder_edit_parity, vae_encode_parity, edit_e2e (health). DiT edit-concat has
   NO numeric pyref (full-DiT OOMs) -- rope verified vs diffusers source + health.
   LATENT BUG FIXED (dit.rs): unused joint-mask buffer binding size must be
   4-byte aligned (odd img+txt joint); `next_multiple_of(2)` on the mask elems.
   REMAINING for a usable endpoint (task 6): Rust-side preprocessing (image load ->
   ViT [N,1176] smart_resize+patchify + VAE [3,1,Hv,Wv] calc-dims resize + edit
   chat-template tokenize w/ image_pad expansion via tokenizers crate), thinfer-app
   ModelId(qwen-image-edit)+defaults+required_files, LocalExecutor wiring, CLI
   `--input-image`, serve ImageSpec+image upload, web UI. Then perf (task 7).
   Reference preprocessing lives in gen_edit_inputs.py + gen_vision_ref._patchify.

   PROGRESS (2026-06-25 PM):
   - CLI WIRED + preprocessing PARITY-PERFECT: thinfer-app/src/preprocess.rs
     (calc_vae_dims 1024^2/mult-32, calc_vit_dims smart_resize/28, CatmullRom bicubic
     ViT + Lanczos VAE, merge-unit patchify, edit-template tokenize + image_pad
     expand). Parity: token_ids EXACT, vit rel 0.000%, vae rel 1.75%. ModelId
     QwenImageEditRapid + ImageKind::QwenImageEdit + ImageRequest.input_image +
     executor::run_qwen_image_edit + CLI --input-image. preprocess_parity test green.
     CLI: `thinfer generate image --model qwen-image-edit-rapid --input-image X
     --prompt Y --width 256 --height 256 --steps 4 --vram-budget 6G --output o.png`.
   - VAE-ENCODE VRAM TILING LANDED + green isolated (wan/vae.rs vae_encode_tile_dims,
     ENC_HALO_PX=128, weight_footprint on WanVaeEncoder, set_transient_reserve;
     mirrors decoder; decoder untouched, single-tile byte-identical). 64x64 green;
     1216x832 (~1MP) under 6GB = 2x2 grid, moments rel .21% / mode 2.13%.
   - EDIT CLI RUNS END-TO-END on 8GB (256x256 + 512x512, 4-step, 6G budget; valid
     non-degenerate PNG). Three thin-hardware blockers fixed (2026-06-25), all were
     Windows-TDR / sticky-reserve, NOT the suspected footprint math:
     1. VAE-encode TDR: encode TILE_MAX 96->48 latent (vae_encode_tile_dims only;
        decode TILE_MAX untouched). At tile 96 the 896px-input 96ch conv3d dispatch
        trips the ~2s watchdog under full-pipeline device load; 48 keeps each tile
        <=640px (~1.1s). vae_encode_parity stays green (small + 1216x832 large).
     2. Sticky transient-reserve LEAK: the VAE encoder set transient_reserve =
        budget-weights (~5.9G) and it was NEVER cleared, so the FOLLOWING DiT denoise
        phase held a phantom 5.9G free on every weight admission -> evicted blocks it
        needed -> OOM. Fix: evict_all_and_free now resets transient_reserve to 0 at
        the phase boundary (the next phase sets its own). residency.rs.
     3. DiT denoise TDR: at the 1MP ref the joint seq is ~4.5k+ tokens; one
        whole-block submit ran ~2s and intermittently TDR'd. Split block_forward into
        block_attn + block_mlp across a submit boundary (carry hs1/eh1 via ws bufs;
        MLP mod signals recomputed from silu_temb). Numerically identical; each
        submit now ~1s. 256@4-step ~550s, 512@4-step similar (joint 5310).
   - SERVE + WEB: DONE. wire::ImageSpec gained `input_image: Option<String>` (base64,
     serialized camelCase `inputImage`); serve/api.rs decodes -> artifact_dir/<id>/
     input_image -> ImageRequest.input_image; validation via ImageRequest::validate
     (edit needs image -> 400; non-edit + image -> 400). Web UI: file-picker row
     (shown when an edit model is picked), qwen-image-edit-rapid in MODELS, buildSpec
     async reads file -> base64 -> spec.inputImage. openapi has inputImage. NOT yet
     deployed to the running server (the .exe is locked by the user's live server).
     USER PREFERENCE (2026-06-25): I deploy serve MYSELF (stop running server ->
     `cargo build -p thinfer-serve --release` -> restart), do NOT hand the user
     manual steps. See [[feedback_user_runs_web_server]] exception.
   - BUDGET = HARD CEILING, PROVEN: 4G budget held vram=4089MiB every denoise block
     (W=2574 + Ws=1515); VAE-encode vram_peak=4096=budget exactly. Scales resident
     DiT-block set to fit (6G: W=4.6G; 4G: W=2.57G). Respects --vram-budget precisely.

## Qwen-Image t2i web exposure -- LANDED + DEPLOYED (2026-06-25)

`qwen-image-rapid` (t2i, text-only, same 4-step CFG-free MMDiT as the edit model)
is reachable from CLI + serve + web UI as its own model id. Read the code; do not
re-spec. What landed:
- `preprocess.rs::tokenize_t2i` (t2i chat template, drop_idx 34, no image-pad);
  parity EXACT vs `tokenize_prompt.py` (`t2i_tokenize_parity`, feature
  `preprocess-parity`, 46 tokens).
- `model.rs`: `ImageModelId::QwenImageRapid` + `ImageKind::QwenImage` (kind/
  manifest/Display/defaults 1024^2/4-step). t2i roles `RUNTIME_ROLES_T2I_Q8` =
  DIT+ENCODER+VAE+TOKENIZER (NO mmproj/preprocessor). The enum drives CLI clap +
  serde + utoipa schema, so the CLI flag + openapi enum updated for free.
- `executor.rs::run_qwen_image` = 3-source union (dit 1:1 + renamed encoder +
  VAE), `tokenize_t2i`, `load(.., edit=false)`, `generate_rgb`.
- `pipeline.rs::load` gained `edit: bool`; the vision tower + edit encoder +
  VAE-encoder now live in an `Option<EditPath>` that a t2i load skips entirely
  (no mmproj download, no extra registered weights). `generate_edit_rgb`
  `.expect`s it. Edit callers (executor, edit_e2e) pass `true`; t2i (executor,
  e2e) pass `false` -- e2e.rs now unions just 3 sources.
- `request.rs`: QwenImage kind rejects `input_image` (falls through the default
  reject arm, same as Z-Image/Ideogram). required_files sources by role.
- web `app.js`: `qwen-image-rapid` in MODELS.image (NOT EDIT_MODELS) + MODEL_STEPS=4.
- PER-STEP PROGRESS (2026-06-25): the qwen pipeline took NO progress callback, so
  the UI jumped TextEncode -> VaeDecode with no "Denoising step i/n". Added
  `pipeline::{ProgressEvent,ProgressFn}` (mirrors z_image/ideogram); `generate_rgb`
  + `generate_edit_rgb` + `generate` now take `ProgressFn` and fire TextEncode /
  Step{i,n} (1-based, in the denoise loop) / VaeDecode. executor `map_qwen` +
  `Some(&progress)` for both run_qwen_image/_edit; the manual sink.stage calls are
  gone. Test callers pass `None`. Redeployed (pid 13824).

DEPLOYED: stopped serve, `cargo build -p thinfer-serve --release`, relaunched
detached (pid 9948, `--config ../../scratch/serve.toml` =
`C:\work\personal\thinfer\scratch\serve.toml`, tls_self_signed, RUST_LOG=
info,thinfer::diag=warn, CWD thinfer/projects). This deploy also carries the prior
pending UI fixes (per-model MODEL_STEPS in app.js; `#input-image-row` display:
contents in style.css). NB rustls is TLS-1.3-only so local PS/.NET/curl clients
can't probe the live endpoint; the browser is the check.

REMAINING: user eyeballs a t2i gen in the browser (the only validation not
machine-runnable -- full-DiT t2i pyref OOMs, component parity is GREEN). Perf:
the `--ref-size` knob below is edit-only; t2i has no ref image.

## Qwen-Image-Edit PERF -- mixed-precision SDPA LANDED + GREEN

DiT denoise = ~90% wall; COMPUTE-BOUND. Measured warm via `dit_perf` bench
(engine-only, loads N blocks like dit_parity, runs forward_multi at a configurable
joint, prints ms/block; knobs QWEN_PERF_{BLOCKS,STEPS,NOISE,REF,TXT,BUDGET}; no pyref,
fast). This bench is THE perf turnaround tool (and the cheap parity tool stays
dit_parity, single-block cached pyref) -- do not iterate perf on the heavy edit_e2e
load+run.

MIXED-PRECISION SDPA -- LANDED + GREEN (2026-06-25; ~2x/block, NO quality loss):
- WHAT: residual + all matmuls stay bf16/Q8_0; ONLY Q/K/V cast bf16->f16 (post-
  rmsnorm/rope = O(1) = f16-safe), run an f16 subgroup SDPA (the O(n^2) long pole),
  cast output f16->bf16. NEW core ops `cast_act::{Bf16ToF16,F16ToBf16}` (packed-word
  casts via built-in pack2x16float/unpack2x16float -- no `enable f16`, no prelude
  mixing; bf16->f16 clamps +-65504). NEW `dit::FastSdpa` (sdpa+2 casts+cl) on
  `QwenImageDitPipelines`, built Some when i8_matmul && backend f16+subgroups;
  `compile(backend,cfgs,i8_matmul)`. block_attn branches: fast path casts jq/jk/jv,
  `scope.sdpa_sg` (has_mask=0, reuses bf16-sized mask binding, sdpa_uniform now
  pub(crate)), casts back; else the bf16 op_sdpa. Wired: pipeline DiT=i8_matmul,
  vision=false (windowed masked op_sdpa), dit_parity/dit_perf default-on (QWEN_NO_I8
  forces bf16), vision_parity=false.
- MEASURED (dit_perf, 6 blocks, 6G): joint 4752 bf16 2123ms/block -> f16 1090ms/block
  (~1.95x); joint 5716 (1MP worst case) 2123->~1306ms. 60-block step ~127s -> ~65s.
- GREEN: dit_parity velocity rel 1.710% (vs 1.766% bf16 -- within noise), block0_hs
  2.321%, temb 0.721%. e2e_health + edit_e2e health GREEN (t2i std 0.689, edit 0.673).
- The block-wide f16 dead-end (below) does NOT apply: only normed Q/K/V go f16, the
  outlier residual stays bf16.

2-SUBMIT BUBBLE -- DISPOSITIVE, do NOT retry (re-measured AFTER SDPA, 2026-06-25):
- Verdict UNCHANGED post-SDPA: the bubble is not worth the VRAM. Re-measured the
  un-split (merge block_attn+block_mlp into ONE scope/submit) at the 1MP joint (~5.7k)
  now that SDPA halved per-block compute: wall UNCHANGED (1330 vs 1306ms/block = within
  run-to-run noise; block stays ~98% GPU-compute-bound) while one scope holds BOTH
  halves' transients at once (BatchScope is a bump allocator -- guards retained for the
  whole scope, NO intra-scope reuse), raising workspace peak ~+350MiB (1989->2339). At
  4G that extra peak forces DiT-block eviction (resident W 2143->1799) = net loss on the
  8GB ceiling. So the worklog's old "un-split = ZERO extra VRAM" premise was WRONG for
  this allocator. KEPT the two-submit TDR split UNCONDITIONALLY (also needed: one whole-
  block submit ~2s = TDR limit on the bf16 path). do-not-retry doc'd in dit.rs block
  loop. (The earlier `submit_deferred` pipelining attempt died the same way: +0.5GiB.)
- BLOCK-WIDE F16 / i8 DP4A: 4.5x faster (f16 alone 2.9x from subgroup SDPA; +i8 matmul
  1.55x) BUT PERCEPTIBLY DEGRADES -- NOT viable. The qwen DiT residual stream has
  large-outlier channels > f16's 65504; under f16 `dit_parity` velocity diverges ~68%
  (slope 0.58) even WITH the upstream per-block clip(-65504,65504) added. f16's 10-bit
  mantissa at magnitude ~10k + the clamped outliers wreck the output. bf16 was a
  load-bearing choice (NOT conservative). The whole i8-DP4A path is dead here because
  `use_dp4a`, `i8_sdpa`, AND subgroup-`sdpa_sg` ALL require ActDtype::F16 block acts
  (block.rs:512/401/536). (ideogram tolerates f16 only because ITS residual stays in
  f16 range.) Tried + reverted: ClampF16 op + per-block clip -- correct mechanically,
  doesn't save quality. Q4_K_M is also moot (streaming is not the bottleneck).

PRIOR (landed, kept): `dit.rs` per-site matmul routing `Site{Qkv,Proj,FfnUp,FfnDown,
Adaln}` + `matmul_site`; `dit::block_cfgs()` = the ONE source of truth for the DiT
block config (bf16, Q8_0 per-site, bf16 adaln), called by pipeline + dit_parity +
dit_perf so it can't drift; `QwenImagePipeline::load(i8_matmul)` threaded from the
ImageRequest/wire/CLI --no-i8-matmul plumbing (now LIVE -- gates FastSdpa, not ignored).

NEXT (perf, remaining): THEN (user OK'd as opt-in, keep 1024^2 default): expose
`--ref-size` capping the VAE ref area in thinfer-app/src/preprocess.rs calc_vae_dims
(512^2 -> ~1024 ref tokens -> ~4x shorter joint, the O(n^2) SDPA shrinks with it). Then
deploy serve myself + hand the user numbers. DO-NOT-degrade: steps(4)/guidance(CFG-free)
untouched; "never PERCEPTIBLY degrade" (tolerance-checked) is the bar.
6. (orig) EDIT PATH (active, 2026-06-25). DECISION LOCKED (user): VAE encode source =
   full `Qwen/Qwen-Image` vae/ safetensors (encoder+decoder, native diffusers
   keys, bf16), used for BOTH decode and encode -- ONE source, pig GGUF dropped.
   Edit (image->image) is the headline feature; --input-image planned, so encode
   is permanent. Perf is NOT a factor in the source choice (encode is one-shot).
   - LANDED + GREEN: manifest VAE role -> safetensors (REPO_VAE=Qwen/Qwen-Image);
     vae.rs pig shim DELETED (vae_source/pig_to_diffusers_vae_key gone); vae_parity
     decode re-validated on safetensors (slope .9989/rel .99%); NEW vae_encode_parity
     (WanVaeEncoder, Wan2.1 non-residual path) GREEN (moments slope 1.003/rel .23%,
     mode slope 1.018/rel 1.93%). gen_vae_ref.py + gen_vae_encode_ref.py load the
     safetensors (strict). Encoder fwd in wan/vae.rs was already implemented (stale
     "follow-up" comment); proven now.
   - SPECS NAILED + persisted in qwen-image-plan.md: full vision-tower op spec
     ("### Vision tower SPEC") + full edit-integration spec ("### Edit integration
     SPEC"). Read THOSE, don't re-derive.
   - IN PROGRESS (delegated subagent): vision tower module qwen_image/vision.rs +
     gen_vision_ref.py + vision_parity.rs. Self-contained (image -> [N/4,3584]).
     2D rope via RopeEmbedder([40,40,0]) + op_rope_halfrot; windowed attn as
     block-diagonal mask over op_sdpa (O(N^2), perf-pass later); GELU merger uses
     tanh GeluF32 (diff within tol). fullatt blocks {7,15,23,31}.
   - NEXT (integration, mine): (a) encoder edit path = embed lookup + scatter
     vision embeds at <|image_pad|> + 3-axis MRoPE position ids (text_encoder.rs
     today is 1-axis collapsed); validate w/ encoder-with-image pyref. (b) DiT edit
     = img stream concat [noise ++ ref_latent_tokens], vid_freqs concat over both
     img_shapes grids, velocity = first noise_seq tokens; 1-block dit-edit parity.
     (c) pipeline edit method (preprocess ref: VAE 32-grid + ViT 28-grid smart_resize;
     encode.mode->normalize->pack; CFG-free). (d) edit e2e health gate (full-DiT
     pyref OOMs -> component parity + visual). 
7. Per-step block streaming + perf; Q4_K_M default; windowed-attn perf. Then CLI
   (--input-image), serve, web endpoint.

## FastWan UniPC sampling path -- LANDED, awaiting visual check

IMPLEMENTED (read the code, do not re-spec). UniPC is now the default FastWan
sampler everywhere (CLI + serve + web UI); DMD stays reachable + is the parity
path. PENDING = the user eyeballs a UniPC clip in the running thinfer-serve web UI
vs the KingNish Space (matched prompt/seed/res). If it looks right, done; if not,
fall back to the GPU upstream-pyref compare. What landed:
- `UniPcConfig::fastwan(steps)` (`wan/unipc.rs`): shift=8.0, 1000 train steps,
  sigma_min 0.001. Verified against the live KingNish `app.py`:
  `UniPCMultistepScheduler.from_config(..., flow_shift=8.0)`, steps default 4 /
  slider 1..=8, `guidance_scale=0` (CFG-free), 896x896 default.
- `VideoSampler { Dmd (default), UniPc{steps} }` on `GenerationParams`; the denoise
  loop in `denoise_with` branches on it. UniPc arm drives the existing `FlowUniPc`,
  no renoise, no guidance; DiT forward byte-identical to DMD (perf unchanged, only
  a tiny host-side latent history added -- no VRAM/residency change). step-diag taps
  stay DMD-only (parity gate uses `VideoSampler::Dmd` explicitly, still GREEN-able).
- Plumbed sampler + steps through `thinfer-app` (`model::VideoSampler`,
  `VIDEO_DEFAULT_STEPS=4`, `VideoRequest`, `wire::VideoSpec`), `serve` (defaults
  UniPC/4), CLI (`--sampler unipc|dmd`, `--steps`), and the web UI (Steps field
  now shown for video, 1..=8 default 4; server defaults the sampler to UniPC).
- Server rebuilt (release) + restarted (default config, 0.0.0.0:8080, embedded web);
  openapi.json carries the new `sampler`/`steps` VideoSpec fields.

ORIGINAL PROBLEM (kept for context): FastWan video looked bad at every res the user
tried (512x288..960x544, tiny AND full VAE). Steps were correct (DMD 3-step
`[1000,757,522]`, matches FastVideo official) and parity is GREEN at 256x256 -- so it
was NOT a kernel/step bug. ROOT CAUSE (found via the reference HF Spaces the user
trusts): they sample the SAME weights with a DIFFERENT denoiser. We used the DMD
re-noise sampler; they use plain UniPC multistep.

CONFIRMED RECIPE (two independent Spaces agree -- `KingNish/wan2-2-fast` app.py and
`rahul7star/Wan2.2-T2V-A14B` app_fast.py, both load `FastVideo/FastWan2.2-TI2V-5B-
FullAttn-Diffusers`): `UniPCMultistepScheduler.from_config(..., flow_shift=8.0)`,
`num_inference_steps=4` (slider 1-8), `guidance_scale=0` (CFG-free). (app_fast also
adds a content LoRA @0.95 -- INCIDENTAL, ignore.) Default res in both is 896x896
square, so square is fine for this model (earlier "avoid square for video" was wrong
for Wan2.2-TI2V).

PLAN (DiT forward stays byte-unchanged; only the sampler around it changes):
- REUSE the existing `FlowUniPc` solver (`wan/unipc.rs`, today only LongLive's). Its
  `UniPcConfig` fields: `sigma_min`, `shift`, `num_train_timesteps`, `sampling_steps`.
  For FastWan: `shift=8.0`, `sampling_steps=4`, `num_train_timesteps=1000`, sigma_min
  as LongLive. NOT touching LongLive.
- Add a NON-AR UniPC denoise loop in `wan/pipeline.rs::generate` as an alternative to
  the `DmdSampler` loop (~line 1046-1149). Per step i: forward DiT at
  `unipc.timestep(i)` -> velocity; `sample = unipc.step(&velocity, &sample)`. NO
  re-noise, NO guidance (CFG-free). Mirrors how `denoise_ar` drives `FlowUniPc`, but
  over the whole latent (no KV window).
- KEEP DMD-3 intact as the parity default so the GREEN gate stays valid. Add a sampler
  choice (DMD | UniPC) + a CONFIGURABLE step count to `VideoRequest`/`wire::VideoSpec`/
  the web UI. DECIDED: step count is a user knob (default 4, range 1-8, per the Spaces),
  NOT a fixed UniPC-4 -- the sparse-distill model is meant to run 3-5 steps. UniPC is
  the serve/UI default sampler for FastWan; DMD-3 stays reachable + is the parity path.
- VALIDATE by eyeballing a UniPC-4 clip vs the Space at matched prompt/seed/res. If it
  looks right, wire it as default; if not, fall back to the GPU upstream-pyref compare.

CARRY-FORWARD (already LIVE in the running `thinfer-serve` -- rebuilt + restarted this
session; do not re-spec, read the code): web UI now has a Size-preset dropdown (trained
aspects, /16 image & /32 video grids, hand-typed = "Custom"), a Quality (VAE) toggle
(tiny default / full), and a Duration(s) field replacing Frames+FPS (sends `durations`).
`default_frames` now = 5s @ model fps (`DEFAULT_DURATION_SECS`), snapping FastWan 121 /
LongLive 125. `ProgressStage::ChunkStep` now serializes camelCase (`numChunks`/`numSteps`)
so the LongLive progress line renders. Server diag probes muted by default
(`info,thinfer::diag=warn`; re-enable via `RUST_LOG`). NB: the user keeps a server
running and toys with it from a browser -- ASK before stopping/restarting it.

## NEXT: `thinfer serve` + OpenAPI + web client

Steps 1-4 LANDED (2026-06-23): the phone goal is reachable. `thinfer-app`
extraction + `thinfer-serve` v1 + `RemoteExecutor`/`--remote` + a server-backed
web UI all build green + clippy-clean; server boots + inits the 5070 worker, the
HTTP surface verified by curl (static UI open + 200, `/jobs*` + openapi gated 401,
valid-token-bad-spec 400 before any download). REMAINING here = the wasm<->http web
toggle + the weights+GPU deferrals below. GOAL (met for server mode): usable from a
phone against a `thinfer-serve` box. Image/video/faceswap over a typesafe OpenAPI
HTTP API + a web page that runs the SAME generation on the server (wasm mode TBD).

DONE THIS PASS (do not re-spec, read the code):
- Wire DTOs live in `thinfer-app::wire` (serde-gated; utoipa `ToSchema` under
  `serve`): `JobSpec`/`{Image,Video,FaceSwap}Spec`, `CreateResponse`, `JobStatus`,
  `JobResult`, `ProgressStage` (+`From<Stage>`/`From<ProgressStage>`),
  `JobStateKind`, `JobEvent` (+`kind`/`is_terminal`). serve re-uses them;
  `spec_into_request` (server-only: artifact path + budget from config) is a free
  fn in `serve::api`. job store/handle/SeqEvent stay server-side.
- `RemoteExecutor` (`thinfer-app::remote`, `remote` feature = reqwest rustls +
  futures-util + serde_json): POST spec -> tail SSE (own fetch-stream frame parser,
  unit-tested) into a `ProgressSink` -> download artifact. Mirrors `LocalExecutor`.
- CLI: `--remote <url>` + `--remote-token` (flattened `RemoteArgs`) on `generate
  image|video`; builds a `JobSpec`, runs via `RemoteExecutor` through the same
  `CliSink` lines. (faceswap remote intentionally absent: server-local path refs.)
- `ServeConfig`: `bind` default `0.0.0.0:8080`; optional `auth_token` (Bearer
  middleware on the `/jobs*`+openapi router only); optional `web_dir`.
- Web UI in `thinfer-serve/web/{index.html,style.css,app.js}`, embedded via
  `include_str!` (self-contained) or served from `web_dir` (dev). Vanilla JS,
  fetch-based SSE so the token rides an `Authorization` header (EventSource can't).
  Mounted outside the auth layer so it can load + prompt for the token. Assets
  served `Cache-Control: no-store` (stale app.js was mis-reading event fields).
- E2E result encryption: browser generates an RSA-OAEP keypair, sends only the
  PUBLIC key in the spec (`public_key`/`publicKey`, optional). Server hybrid-
  encrypts the artifact at rest (random AES-256-GCM key, RSA-OAEP-wrapped;
  `serve::crypto`, ring provider), serves opaque bytes; browser unwraps + AES-
  decrypts to an in-memory blob. No key => plaintext (CLI path). WebCrypto needs
  a SECURE CONTEXT (https/localhost); app.js warns + falls back to plaintext on
  insecure http.
- Result is DELETE-ON-FETCH: `GET /jobs/{id}/result` reads bytes, removes the job
  dir, returns them; second fetch 404s. Browser holds the only lasting copy.
- Opt-in HTTPS for the secure context over LAN: `tls_self_signed` (rcgen self-
  signed at startup, SANs = localhost+127.0.0.1+auto-detected LAN IP+`tls_sans`)
  or BYO `tls_cert`/`tls_key`. axum-server+rustls(ring), no aws-lc C build.
- MP4 faststart (`codec::faststart`): moov relocated before mdat, stco/co64 chunk
  offsets patched; web/strict-player playback + seeking. Unit-tested.
- Web result UI: explicit Download link with a `thinfer.{png,mp4}` filename (a
  bare blob: save dropped the extension -> OS "can't play").

STATE OF THE TREE:
- `thinfer-app` (new lib): `model` (id enums + defaults + frame grid + manifest),
  `request` (`JobRequest` + per-modality structs + shot-plan + `required_files`),
  `progress` (`Stage` + `ProgressSink`), `download`, `codec` (mp4/png-frames/faceswap
  stream), `executor::LocalExecutor`, `config` (`BackendConfig`/budget/mem). Features:
  `cli` (clap ValueEnum), `serde`, `serve` (serde+utoipa ToSchema on the id enums).
- `thinfer-cli`: thin clap adapters -> app requests; CLI keeps env->BackendConfig,
  consent prompt, decile download logging, stamped `CliSink`, mem rollup. Behavior
  preserved (--help defaults match; 10 shot-plan tests moved to app, green).
- `thinfer-serve` (new bin): axum + utoipa. `POST /jobs` (image/video queue; faceswap
  409-if-busy), `GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE, in-mem log replay +
  Last-Event-ID), `GET /jobs/{id}/result` (streams from disk), `POST /jobs/{id}/cancel`,
  `GET /openapi.json` + `--emit-openapi`. Workers = OS threads w/ current-thread
  runtime (avoids Send bound on `!Send` generate futures), one `LocalExecutor` each
  (default 1). In-mem job metadata, on-disk artifacts under `artifact_dir/<id>/`, no
  DB. TOML `ServeConfig`.

DEFERRED (do these in step 2.5 / when running with weights):
- Mid-generate cancel: NOT wired (would touch the shipped z_image/wan `generate`
  signatures = DO-NOT-DISTURB FastWan). v1 cancel only dequeues a QUEUED job; a
  running job finishes. Add a cancel token (or make `ProgressFn` return ControlFlow)
  with a warm before/after, with sign-off.
- serve==CLI byte-parity test: same request via `--remote` vs local -> identical
  bytes (deterministic, fixed seed). The `--remote` vehicle now exists; needs
  weights+GPU, add under `wan-e2e`-style gating. (HTTP surface itself is curl-checked
  without weights; the SSE frame parser has unit tests in `remote.rs`.)
- Disk-backed SSE ring buffer (survive restart) -- v1 keeps the event log in memory.
- Server video is MP4-only (png-frames is a CLI debug format).
- 422 vs 400: a well-formed-JSON-but-wrong-shape body returns axum's 422 (handler's
  own semantic validation returns 400). Fine; revisit if a client needs 400.

CORE ABSTRACTION -- one `Job`, one progress vocabulary, one `JobExecutor` trait,
consumed everywhere: CLI (local/remote), serve (local pool), web (wasm/http).
- `JobExecutor` impls in `thinfer-app`: `LocalExecutor` (download/load/generate on
  this machine) and `RemoteExecutor` (HTTP client to a `thinfer-serve`: POST job,
  stream SSE, render the SAME stderr progress lines, download result). Feature-gate
  `RemoteExecutor` behind a `remote` feature (reqwest/SSE deps).
- `thinfer-cli --remote <url>` selects `RemoteExecutor`; args parse into the same
  request struct either way. Exact mirror of the web app's wasm<->http toggle.

CRATE LAYOUT. `thinfer-app` is a NEW lib (not a rename): holds the `Job`, request
types, progress vocabulary, `JobExecutor` trait + `LocalExecutor`, and the
orchestration currently duplicated across `cmd/generate/{image,video,faceswap}.rs`
(validate -> parse budget -> resolve variant -> resolve/download -> open mmap
openers -> init backend -> generate -> encode). Built on `thinfer-native` +
`thinfer-models`. `thinfer-cli` shrinks to a clap adapter; `thinfer-serve` = axum
adapter. Future native consumers (`-desktop`/`-python`/`-android`) reuse `app` too.
- Layering: core < models (dep-clean) < native < app < {cli, serve, ...}.
- `thinfer-web` (wasm) is the ONE exception -- own browser substrate (fetch/JS file
  handles, no tokio/mmap), CANNOT dep on `app`; it already reimplements the load
  dance and calls `thinfer-models` directly. Cross-language sharing is via OpenAPI
  (generated TS / future python clients), NOT a shared Rust types crate -- so types
  stay in `app`, no separate types crate until a real second consumer needs it.

API SURFACE (no `/v1`, no models-list endpoint; model id is an OpenAPI enum in the
request schema so it cannot drift):
- `POST /jobs` -> `{id, queue_position}`. Large-INPUT jobs (faceswap) return `409`
  if the worker is busy instead of queuing; small-input jobs (image, video t2v)
  queue. Per-request-type rule, single endpoint -- faceswap is still a job.
- `GET /jobs/{id}` -> status snapshot (poll fallback).
- `GET /jobs/{id}/events` -> SSE: `queued{position}/started/progress{phase,i,n,eta}
  /done{result_url}/error`. Disk-backed ring buffer + `Last-Event-ID` reconnect
  (jobs run minutes; clients drop). This is a THIRD sink for the already-structured
  `ProgressEvent`/`ProgressFn` (NOT stdout scraping) alongside CLI stderr + web JS
  fn. Unify the per-model events (z-image TextEncode/Step/VaeDecode vs video/
  faceswap phases) into one vocabulary.
- `GET /jobs/{id}/result` -> streams the artifact from disk (content-type + range;
  NEVER base64 a video into JSON).
- `POST /jobs/{id}/cancel` -> cancel/dequeue; job SURVIVES as `cancelled` (cancel !=
  delete). `DELETE /jobs/{id}` optional/skipped for v1 (TTL sweep handles cleanup).
- Cancellation needs a cancel token threaded into the `generate` signature
  (alongside `progress`, checked between steps) -- decide NOW, it touches the sig.

LARGE DATA / STORAGE: disk for blobs, in-memory for metadata, NO DB. Inputs
(faceswap = disk path-ref, gated to localhost/allowlist -- arbitrary-file-read
footgun), outputs, and the SSE ring buffer live under an artifact dir keyed by job
id. Metadata (state, params, queue pos, progress snapshot, result path, error) in an
in-memory map. ACCEPTED tradeoff: restart loses in-flight/queued metadata (artifacts
orphan -> TTL sweep); fine for single-user. Retain-with-TTL + max-bytes LRU (not
stream-once) so the page can re-download / show a small gallery.

CONCURRENCY: worker POOL of 1..N (one per device), CONFIGURABLE (default 1). Local
N=1 -- DiT denoise is at the matmul ceiling, compute-bound, batch=1 saturates the
28-SM mobile 5070, so concurrency = pure latency tax. Multi-GPU = trivial N workers.
Intra-GPU batching (the vLLM analogy) is NOT v1: vLLM batches bandwidth-bound LLM
DECODE; diffusion is compute-bound + lockstep over a fixed latent (opposite shape),
so batching only pays on UNDER-saturated big GPUs and needs dim/step-compatible
splicing. Job carries dims/steps/model so it is addable later with no schema change.

CONFIG: TOML file for `thinfer-serve` (deployment: bind addr, worker count / device
list, artifact dir, retention TTL, auth token, budgets) + a few flag overrides.
SEPARATE from generation defaults (steps/dims/guidance), which move into the model
registry (`ModelId::defaults()`, the existing `image.rs` TODO) so CLI + API read one
source and cannot drift.

WEB APP (served by `thinfer-serve`): NOT circular at the cargo graph -- `serve` does
not dep the `thinfer-web` crate. Seam is `openapi.json` (emit via a small bin/test,
no GPU). The pnpm/TS app builds against `openapi-typescript`/`openapi-fetch` (server
mode) + the `thinfer-web` wasm npm pkg (wasm mode); the wasm<->http toggle is a
TS-level `Executor` swap behind one interface. Embed built assets behind a feature
flag: release embeds (self-contained binary), dev serves from a dir (edit-reload, no
Rust rebuild). CAVEATS for the toggle: it is per-CAPABILITY -- wasm does image only
(10GB video won't fit an 8GB browser budget; faceswap is native-tool-heavy), so
video/faceswap are server-only in the UI; and output format diverges (server MP4 via
native openh264, wasm PNG-frames / WebCodecs), so the result type carries its format.

STACK: axum + `utoipa`/`utoipa-axum` (OpenAPI 3.1, OpenApiRouter = route+docs), Scalar
docs UI; GPU on dedicated worker thread(s) / spawn_blocking. Feature-gate the utoipa
`ToSchema` derives so they never reach a non-serve build.

BUILD ORDER:
1. DONE -- `thinfer-app` + `ModelId::defaults()` + `thinfer-cli` re-pointed (cancel
   token deferred, see DEFERRED).
2. DONE (v1) -- `thinfer-serve` (TOML config, worker pool, queue, SSE w/ in-mem log,
   disk artifacts, `openapi.json` + `--emit-openapi`). Disk ring buffer deferred.
3. DONE -- DTO move to `thinfer-app::wire` + `RemoteExecutor` + `thinfer-cli
   --remote`/`--remote-token`. See DONE THIS PASS above.
4. DONE (server mode) -- server-backed web UI served by `thinfer-serve` (embedded /
   `web_dir`). The wasm<->http toggle is NOT built (server-only by decision); when
   adding it, the seam is a TS-level `Executor` swap + the `thinfer-web` wasm pkg
   (image-only in-browser), and the user runs the web dev-server + reports browser
   results.

## NEXT (active): LongLive-2.0-5B AR perf

Read the code, not a re-spec (`WanDit::forward_ar`, `WanModel::denoise_ar`/
`generate_ar`, `wan/kv_cache.rs`, `wan/unipc.rs`).

LongLive runs many forwards (chunks x [4 UniPC steps + 1 clean recache]) vs
FastWan's 3, so it is SLOWER at a fixed length -- the O(N)-streaming win is
LENGTH/VRAM-bound generation, not shorter wall. Warm (page-cache hot) the AR
forward at 576 is ~86% compute-bound (matmul ceiling, same ops as FastWan); the
~15% "idle" seen cold is a disk artifact, NOT recoverable per-block streaming.
RE-MEASURE WARM at 576 before chasing any lever. Wins left (quality-neutral, exact),
value order:
- Upload the window prefix ONCE per chunk, not per forward (identical across a
  chunk's forwards; cuts HtoD traffic).
- Activation-tile the AR self-attn (workspace ~= budget at 576; mirror
  `forward_block_tiled`) -- buys prefetch/residency headroom.
- Cache cross-attn text K/V across a chunk's forwards (same prompt; we have no
  cudagraph so it is free).
- Skip the head (proj_out + unpatchify) on the clean recache pass (velocity
  discarded; small).
DEAD END: prefetch-overlap in `forward_ar` (mirror `forward`'s `join!`) -- TRIED +
REVERTED, ~1% warm (noise). Do not re-add without a warm before/after.

Also OPEN: multi-shot is only health-tested. A multi-shot pyref byte-parity (extend
`gen_longlive_video_e2e_ref.py` to the multi-prompt block list) is the durable proof
if multi-shot quality must be guaranteed; the `zero_for_scene_cut` path is unexercised.

AR-loop invariants (do not relitigate):
- Within a chunk, `current_start` is CONSTANT across the 4 UniPC steps + clean pass;
  each forward recomputes the chunk K/V at the same tail slot. Only the timestep-0
  clean pass K/V are committed (survive into future chunks).
- Convert frames<->tokens with `frame_seq_len = pph*ppw` everywhere.
- ABSOLUTE-position temporal RoPE: q + chunk-k rotate at `chunk_start_frame =
  current_start/frame_seq_len`; cached prefix-k stored already-roped (no re-rotation
  at attention time); `temporal_offset = shot_index * 8` folds into the frame id
  (`rope3d.rs lookup_temporal`).
- Single-prompt T2V modulation collapses to FastWan's scalar-t broadcast, so
  forward_ar reuses the FastWan embedder/modulation unchanged; ONLY self-attn differs.

GROUND TRUTH IS THE CLONE (`third-party/LongLive`, NVlabs/LongLive): AR loop
`pipeline/causal_diffusion_inference.py`, self-attn/cache `wan_5b/modules/
causal_model.py`, sampler `wan_5b/utils/fm_solvers_unipc.py`, `configs/inference.yaml`
(chunk 8 / window 32 / sink 8 / 4 steps / shift 5.0). IGNORE nvfp4. Weights: HF
`Efficient-Large-Model/LongLive-2.0-5B` `model_bf16.pt` (10GB, 825 bf16 tensors,
complete merged DiT, LoRA pre-folded); umT5+VAE reused from FastWan.

LESSON -- LongLive AR parity (RESOLVED, do NOT re-open the op-hunt): the engine is
arithmetically faithful per-op AND per-forward (`vel_c0s0` within band; AR path
bit-identical to GREEN FastWan). The only gap is AR-depth compounding of the
16-bit-vs-bf16-locked-pyref per-forward rounding (~2%/forward in large-outlier
residual channels, blocks 15-25) across many forwards -- NOT a code bug, not
precision-fixable (upstream hardcodes bf16 SDPA). Tolerated via the two-tier band in
`longlive_parity.rs` (tight `TOL_LATENT` on the single-forward `vel_c0s0`; loose
gross-regression floor on the AR-compounded tensors).

DECISIONS LOCKED: bf16 full precision, NO quant (NVFP4 skipped); runtime `.pt`
ingestion (no build step / no on-disk dup -- footprint first-class); FastWan `forward`
stays byte-unchanged (AR behavior lives in the AR path only); track here (no
longlive-plan.md).

## Lessons / dead-ends (do not retry)

- **DiT denoise is at the WGSL matmul ceiling.** 100% matmul+sdpa GPU time;
  ~2.5-3.4 TFLOP/s = ~20-25% of the f16 issue ceiling, latency/occupancy bound (NOT
  bandwidth), so bigger tiles BACKFIRE on the 28-SM mobile 5070. Weight-ONLY quant
  does not help (dequant->bf16, same FLOPs). Only real levers are backend-level:
  tensor-core (WGSL/naga expose no WMMA -- likely blocked) or i8 DP4A (done).
  REVERTED: tile_b per-kk2 hoist, bk 16->64. Measure via e2e
  (`THINFER_E2E_SKIP_PYREF=1 THINFER_E2E_FRAMES=13`, read `gpu_ms by pipeline`), NOT
  microbench.
- **i8 DP4A matmul (default on; `--no-i8-matmul` = bf16 reference).** Transcodes
  ffn_up + self-attn-qkv weights bf16->Q8_0 at load, routes the i8xi8 `dot4I8Packed`
  path; ~5-6x those ops, ~-30% wall, quality-neutral (parity GREEN both models; i8
  error < the f16-vs-bf16 per-forward gap because these A-sides are normed/modulated,
  no outliers). NOT i8'd, stays bf16: proj + ffn_down (A-side attention-out/gelu has
  ~16k outliers that per-32 act-quant crushes); cross-attn qkv (K/V project from
  UN-NORMED umT5 text, i8 acts overflow f16). The qkv site is split into
  `matmul_qkv_self` (i8) vs `matmul_qkv` (bf16) in the shared block configs (== each
  other unless pinned, so FastWan/Z-Image/umT5 byte-identical).
- **DiT activation-tiling tier** (`wan/dit_block.rs`/`dit.rs`): pass A (row-tiled
  norm1/qkv/qk-norm/rope) -> global self-SDPA barrier -> pass B (row-tiled
  o-proj/residual/cross-attn/FFN), each its own submit. `DIT_TILE_ROWS=1024`, engages
  above one tile. Bit-exact; bounds VRAM (flash `sdpa_sg`, no materialized [n,n]).
- **VAE decode is conv-GPU bound** (~95% pure conv time; authoritative metric is
  `gpu_disp_ms` for `vae_decode`, NOT nvidia-smi util). Conv tiles tuned (`wan/vae.rs`
  128x96x16) -- implicit-GEMM convs are bandwidth-bound, bm=128 halves weight reads,
  48 acc/256 threads is the occupancy knee; bit-exact (f32 accum). DO NOT retry the
  conv3d im2col loop-invariant-div hoist (REVERTED, 262->301s).
- **VAE decode VRAM tiling**: live set `FIXED(tout) + area*PER_AREA(tout)` per spatial
  tile, sized from `budget - reserve` (reserve queried, not fractional); DiT weights
  freed before VAE (`evict_all_and_free`). Constants in `vae_tile_dims` (recalibrate
  if the decoder graph changes).
- **Tiny VAE (LightTAE) is the `--vae` default** (`--vae full` = parity path).
  Temporal-chunk decode tiling: `memcat` ping-pong carries each chunk's trailing frame
  (causal depth 1, no halo); a clip that fits = one chunk, bit-identical to untiled.
  Durable test `THINFER_E2E_TINY=1`.

## Carry-forward gotchas (Wan-general; reused by LongLive)

- umT5 MUST run bf16 acts. Its residual stream grows past f16's 65504 by block ~20 ->
  inf -> NaN in `final_layer_norm` -> token-uniform "washed blob" (magnitude is
  PROMPT-dependent, so f16 blew up only on some prompts). `pipeline.rs::load_with_act`
  compiles umT5 bf16; DiT stays f16 (the umT5->DiT seam is host-f32 readback). Check
  non-finite NOT just NaN (`inf.is_nan()` is false).
- RoPE3D is interleaved-pair, NOT half-rot (opposite of Qwen3). Freqs MUST pack to the
  act dtype (`freqs_upload_bytes`): f32 freqs into an f16 kernel -> inf -> NaN.
- bf16->f16 reinterpret class: broadcast vectors that are STORED WEIGHTS
  (scale_shift_table, norm2 affine) use a `weight_dtype`-keyed op (`bcast_add`/
  `bcast_mul`), not an act-scale op. New broadcast site: check weight vs act.
- umT5 even-pads odd token counts by duplicating EOS; that pad key MUST be masked
  (`wan/umt5.rs`) or bidirectional attention double-counts it.
- DiT driver takes `text` as host f32 `[text_seq, text_dim]` (umT5 readback + reupload),
  zero-padded, no cross-attn mask.
- Shared helpers: Wan DiT reaches into `z_image::{block, embedders, rope_embedder,
  seq}`. Consider extracting a `thinfer-models` common module as the family grows.
- Video: per-frame PNG sequence / tiled contact sheet for staging; MP4/WebM in the CLI
  only (openh264).

## Running the e2e / measuring

Card is RTX 5070 Laptop (8GB); keep budgets <8GB (8GB OOMs the device).
- FastWan parity gate (needs HF bundle + `uv`): `THINFER_TRACE=1
  THINFER_POWER_PREF=high THINFER_E2E_BUDGET_GB=6 THINFER_E2E_WIDTH=256
  THINFER_E2E_HEIGHT=256 THINFER_E2E_PNG_DIR=<dir> cargo test -p thinfer-conformance
  --features wan-e2e --release video_e2e -- --nocapture --test-threads=1`. Perf/trace
  only: add `THINFER_E2E_SKIP_PYREF=1`. NEVER run the fp32 pyref above tiny dims
  (~30GB host).
- LongLive parity: `longlive_parity` (use 256x256, NOT the 128 default which is
  pathological); LongLive e2e: `longlive_e2e`. Both `--features wan-e2e`.
- CLI full run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate video
  --prompt ... --width 576 --height 576 --vram-budget 5G --ram-budget 5G
  --download-as-needed --output out.mp4`. Rollup + `[mem]` at process EXIT; read per-op
  `gpu_ms by pipeline` to localize perf. Inspect pixels via `--output-format png-frames`
  or ffmpeg.
