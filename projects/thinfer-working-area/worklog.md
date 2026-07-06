# Worklog

Forward-looking only. Git history is the changelog, the code is the record.
Engine-wide design: `plan-details.md`. Per-model plans are separate files (see
Status). Scratch is ephemeral; nothing here depends on a scratch file.

## NOW: HyperSwap (ONNX face-swap) BATCHING -- the big perf lever (2026-07-06)

User wants the LARGE-effort/LARGE-gain path. Profiled the ONNX face-swap
(`generate face-swap`, SCRFD -> ArcFace -> HyperSwap -> paste): per frame (1 face)
= detect ~15ms + warp ~2ms + hyperswap_run ~136ms + paste ~13ms => ~163ms/frame.
detect (640^2) + hyperswap (256^2) are FIXED-SIZE (independent of video res); only
warp/paste scale. So 30s@720p@30fps ~= 900 frames ~= 2.5 min today.

ROOT CAUSE (measured, not guessed): hyperswap is ~510 SERIAL dependent GPU
dispatches on a tiny 256^2 GAN (Conv x98 [already on the tuned implicit-GEMM path],
Mul x86, Add x73, Relu x71, Gemm x42, Sigmoid x22, Sub x22, InstanceNorm x16,
LeakyRelu x13, ...). Only ~2 GFLOP-class compute but 136ms => LAUNCH-LATENCY-BOUND
on the serial dispatch chain (NOT compute, NOT memory: single-submit 136ms vs
per-submit 194ms, so submit overhead is small; it is GPU-side per-dispatch launch/
sync on a dependency chain). Convs already tuned; the graph Expands are shape-
subgraph (const-folded, never on GPU) -- an Expand-elision attempt gave 0% and was
reverted. There is NO small safe lever.

THE LEVER = BATCH B frames through the GAN (amortize the fixed per-dispatch latency
across B => ~B x throughput; target 30s@720p 2.5min -> ~30-45s). BLOCKER already
pinpointed: the model is baked at batch=1 -- all 43 Reshape targets hardcode
`[1,C,1,1]` (e.g. `[1,1024,1,1]`, `[1,-1,1,1]`), so feeding `[B,3,256,256]` breaks
at those reshapes. PLAN:
1. GRAPH REWRITE (thinfer-core/src/onnx): at load, rewrite the 43 batch-1 Reshape
   target constants so batch flows (set the leading dim to -1 where the shape has
   no other -1; where it is already `[1,-1,1,1]` set the lead to explicit B). Verify
   plan() infers `[B,...]` end to end and the source embedding input becomes
   `[B,512]` (tile the SAME source emb B times). SCRFD detect also batch-1 -- either
   keep detect per-frame (only ~15ms) or batch it too (bigger win, more work).
2. PIPELINE RESTRUCTURE (codec.rs swap_video_streaming): buffer B decoded frames,
   detect each, batch the B aligned 256-crops into one `[B,3,256,256]` hyperswap
   run, then paste each result back. Keep RAM bounded (B is small, e.g. 4-8).
3. faceswap/mod.rs: `swap_frames_batched(&[Image], emb) -> Vec<Image>`; OnnxModel
   run with B>1 (executor already batch-agnostic for conv/elementwise/IN; the
   Reshape rewrite is what unblocks it).
GATE: byte-identical (or within fp noise) output vs the current per-frame path at
B=1; measure hyperswap ms/frame at B=4/8 (should fall ~linearly until the GPU
saturates); report the real 30s/720p number. RISK: the Reshape rewrite must be
exactly right or output corrupts -- add a unit test on a synthetic batch-1 Reshape
graph + an e2e A/B.

--- DreamID-V + face-swap: LANDED this session (uncommitted -> committing now) ---
- .pth LOADS DIRECTLY (no offline convert): manifest DIT_DREAMIDV -> repo-root
  `dreamidv_faster.pth` (was wrong subdir path), opened via PytorchSource +
  dit_gguf_renames(30) in open_dreamidv_source. Re-verified parity on the .pth
  path: Stage A rel_rmse 9.47e-4/cosine 0.99999314 (identical to the old
  safetensors path), Stage B cosine 0.99997. context.pth is NOT in the HF weights
  repo (404; it lives in the code repo) -> embedded the [4,4096] bf16 constant
  in-tree (`wan/dreamidv_context.bf16` + `wan::dreamidv::baked_context()`), 32 KiB.
  Deleted the executor dreamidv_dir/load_context_safetensors hack. Conformance
  dreamidv_e2e updated to the .pth + embedded context.
- FACE IMAGE RAM-ONLY: new `request::FaceImage(Vec<u8>)` (redacting Debug) through
  VideoRequest/FaceSwapRequest + CLI + serve; serve no longer writes the face to
  the job dir. `codec::load_image_bytes`.
- AUDIO-TOLERANT DEMUX: `codec::strip_audio_traks` rewrites each `soun` trak to a
  `free` box IN PLACE (moov length preserved -> mdat/stco untouched, faststart-
  safe), applied to both decode sites, unit-tested.
- DreamID PERF (frozen as a ~5s/480p tool -- 30s/720p is out of its envelope:
  full spatiotemporal O(tokens^2) attention, tokens ~= latent_frames * grid, so
  30s@720p ~= 200x the tokens of a short 480p clip => hundreds of hours + OOM):
  coopmat/fast_sdpa wired on (quality-neutral, parity identical); i8 DP4A on
  qkv_self+ffn_up (WanI8Sites) is ~24% faster BUT visibly degrades the swapped
  face (A/B: mean 2.6%/worst 5.2% pixel diff, red-smear artifact) -> DEFAULT OFF
  (executor passes i8_matmul=false), capability retained on DreamIdvPipeline::load
  for A/B. i8 stage-A parity with i8 on = 4.1e-2 (vs 9.5e-4 bf16). VAE 480p OOM
  self-recovers via the existing wan-VAE budget-reseed (tile 52->43), so full 480p
  runs; item-4 Q8/tiny-VAE footprint work NOT pursued (path frozen).

Full plan: `dreamidv-plan.md`. User pivoted here (on AUTO; deliver in serve).
Three deliverables: (1) expose existing ONNX face-swap in WEB (CLI done); (2)
port DreamID-V-Wan-1.3B-Faster (diffusion video face swap, 16-step, FastWan-speed
class); (3) encrypted video upload from web (stream/RAM; disk spill -> vault
AES-GCM). Usual gates: e2e parity, ram/vram budgets, best quality.

ARCH FULLY REVERSE-ENGINEERED (dreamidv-plan.md has the detail). Faster variant =
STOCK Wan2.1-1.3B T2V DiT (dim1536/ffn8960/L30/H12) + 2 deltas: in_dim 48
(noise16 + video16 + mask16 channel-cat) and a ref_conv Conv2d(16->1536,2x2) that
patchifies the source-face latent into PREFIX tokens (grid F->F+1, stripped after
blocks). No pose/projector/CLIP in faster. Text context is BAKED (context.pth,
4.2MB umT5 embed of "chang face") -> NO live text encoder. Reuse: Wan2.1 VAE
(wan2_1() z16, encoder EXISTS) + UniPC (shift5) + Wan DiT blocks/rope3d, all
present. License Apache-2.0 (commercial OK). Weights dreamidv_faster.pth 5.69GB
DOWNLOADING (scratch/dreamidv_dl.log). .pth is pickle -> convert to safetensors
once (no runtime pickle reader).

REUSE MAP (from code sweep): WanDitConfig @ wan/dit_block.rs:101 (add
wan2_1_1_3b(), no 1.3B exists yet); WanVariant @ wan/pipeline.rs (add dreamid_v);
VaeConfig::wan2_1() + register_encoder @ wan/vae.rs; VAE_WAN21 manifest role;
run_video/run_wan @ executor.rs; VideoModelId @ model.rs (input_image field
exists but REJECTED on Wan path -> wire it); upload = base64-in-JSON only
(spec_into_request @ serve/api.rs writes to job dir; extend for video); vault
derive_cipher/seal/unseal @ app/vault.rs (make pub for at-rest video encrypt).

BUILD ORDER (parity before perf): inspect+convert .pth -> config+variant+loader
-> DiT ref_conv/48ch delta (gated Option, other Wan models untouched) -> pipeline
(VAE-enc video/mask/image, image-CFG 2fwd/step, UniPC) -> e2e parity gate
(committed a_girl clip+precomputed mask+ref img fixtures, tiny dims) -> DWPose
port (yolox+dwpose ONNX, face-hull mask; DEFERRED, precomputed mask unblocks
parity) -> app wiring + encrypted upload + face-swap web -> serve deploy.

LANDED this session (uncommitted, thinfer-models compiles GREEN, exit 0):
- WEIGHTS: dreamidv_faster.pth (5.69GB, 827 tensors, ALL FP32, 1.42B) downloaded
  + converted -> scratch/dreamidv/{dit.safetensors 5.68GB, context.safetensors}.
  Full tensor manifest in dreamidv-plan.md (block names = ORIGINAL-Wan). context
  = baked "chang face" umT5 embed (~[4,4096]). Converter: scratch/dreamidv_convert.py.
- CONFIG: WanDitConfig::dreamid_v() (dim1536/ffn8960/L30/H12, in_ch 48, out_ch 16,
  ref_conv true) + new `ref_conv: bool` field (all other constructors false).
  dit_block.rs.
- LOADER FULLY SOLVED: reuse `dit_gguf_renames(30)` (source.rs; original-Wan ->
  diffusers, NO prefix unlike longlive) + patch_embedding/ref_conv ride
  passthrough. LoadedWanDitHandles gained `ref_conv: Option<LinearBiasHandles>`
  (dit.rs); register_wan_dit_handles registers it via register_conv_as_linear_bias
  when cfg.ref_conv (loader.rs). DreamID DiT source = just
  `RenamedSource::with_passthrough(ShardedSafetensorsSource(dit.safetensors),
  dit_gguf_renames(30))` (no union; VAE loads separately from VAE_WAN21, NO umT5).

DWPose port: DONE + VALIDATED (uncommitted). New `faceswap/dwpose.rs`:
`face_mask_video(backend, yolox_onnx, dwpose_onnx, frames) -> Vec<Image>` +
`DwPose::{load, face_mask}`. yolox_l (person det) + dw-ll_ucoco_384 (RTMPose
133-kpt SimCC) through OUR onnx executor; face pts [23:91] -> monotone-chain
convex hull -> scanline fill -> separable 15x15 max-dilate. Extended the ONNX
executor (thinfer-core/src/onnx/{kernels,shape,exec}.rs) with Slice/Split/MatMul/
GlobalAveragePool/ReduceSum/Clip/Sqrt/HardSigmoid + N-way non-unit Concat + a
const-Identity fold fix. Validated vs onnxruntime: yolox rel 2.3e-5, dw-ll rel
~1e-5 (133/133 kpt argmax bit-exact), e2e mask IoU 0.987 vs the repo's reference
a_girl mask. Tests: thinfer-core/tests/dwpose_ops.rs + conformance/tests/
dwpose_onnx.rs (faceswap-e2e). onnx models in HF cache; goldens scratch/golden.
This is the live-path preprocessing (uploaded target video -> face-mask video ->
VAE-encode -> mask conditioning latent).

CORE DONE (steps 1-4, uncommitted; WHOLE WORKSPACE clippy-clean exit 0 with
dreamidv-e2e+faceswap-e2e features; both subagents integrated):
- DiT surgery (wan/dit.rs, gated cfg.ref_conv): WanDitInputs.img_ref; assemble
  sizes block over n_tok+ref_rows (ref_rows=pph*ppw); forward writes video tokens
  at offset ref_rows, ref_conv patchifies img_ref spatially -> rows 0..ref_rows,
  rope lookup(ppf+1,pph,ppw), head/unpatchify slice x[ref_rows..]. Tiling off for
  dreamid. OTHER WAN MODELS BIT-IDENTICAL (FastWan video_e2e PASSES post-change;
  ref_rows=0 -> byte-identical path). All 827 tensors load, 0 UnknownWeight.
- wan/source.rs open_dreamidv_source (renamed safetensors + Wan2.1 VAE union);
  pipeline.rs WanVariant::dreamid_v(); unipc.rs UniPcConfig::dreamid_v (shift5);
  manifest.rs REPO_DREAMIDV + roles DIT_DREAMIDV/CONTEXT_DREAMIDV + VAE_WAN21.
- NEW wan/dreamidv.rs: DreamIdvPipeline::{load, generate}. Host preprocess
  (NaResize/DivisibleCrop16/Normalize0.5 video+image, none mask, white-pad src)
  -> Wan VAE encode (WanVaeEncoder::encode EXISTS; (mu-mean)/std) -> 48ch
  cat(noise,y) -> image-CFG denoise (pos_tiv + 4.0*(pos_tiv-pos_tv), 2 fwd/step,
  UniPC) -> VAE decode. conformance dreamidv-e2e feature + tests/wan/dreamidv_e2e.rs.
- HEALTH (128^2-area,5f,2step): all stages finite+non-trivial (video_lat std
  0.58, denoised std 0.69 range[-2.5,3.4], rgb std 0.36 in[-1,1]); PNGs
  scratch/dreamidv/out/. NOT parity (see below).

PARITY CAVEATS to resolve (from the core agent): guide scale = 4.0 (CONFIRMED
correct: generate_dreamidv_faster.py CLI default --sample_guide_scale_img 4.0,
NOT the 5.0 fn-signature default); preprocessing resampling is BILINEAR in our
engine vs torch bicubic(NaResize)/LANCZOS(src fit) -> decouple in parity (feed
shared VAE latents to isolate the DiT); tiled DiT path off (perf follow-up).

PARITY: PASS (verdict mine, 2026-07-05). Stage A DiT single-forward vs torch fp32
(our PRODUCTION bf16/f16 config): rel_rmse 9.47e-4, slope 0.999709, cosine
0.99999314, mean/std match 4 decimals; PER-BLOCK flat across all 30 (rel 2e-5..
1.7e-4, no divergence onset) => ref_conv prefix-token surgery is numerically
correct, the ~1e-3 is pure 16-bit rounding. .pth loads strict 0 missing/0
unexpected (config+rename map exact). Stage B VAE-encode: cosine 0.99997, slope
1.012 (~1.2% latent scale, within 16-bit conv noise on the reused encoder;
watch at eyeball, non-blocking). Harness: scratch/dreamidv_parity_ref.py +
conformance dreamidv_{dit,vae}_parity tests + additive read-only accessors
DreamIdvPipeline::{dit_forward_parity, vae_encode_parity} (reuse existing paths,
no model-logic change; mirror LongLive diag hooks). Run: THINFER_DREAMIDV_PARITY=1
cargo test -p thinfer-conformance --features dreamidv-e2e --release
dreamidv_dit_parity dreamidv_vae_parity.

NEXT: (C) real-dim e2e eyeball (mine, in flight) -> then app/serve/web wiring +
encrypted upload + face-swap web (step 5) -> serve deploy (step 6, ASK first).

APP WIRING DONE + SMOKE-TESTED (uncommitted; workspace clippy-clean). Full
CLI/serve/web + encrypted upload wired (see below). CLI smoke test of the REAL
executor path (decode -> LIVE dwpose mask -> pipeline -> mp4) PASSED: 13f 432x288
16step in 163.9s incl live DWPose + cold transcode. Crypto reviewed + sound
(ephemeral 256-bit key, video >512MiB spills AES-256-GCM to job dir, plaintext
dropped, decrypt-on-read, key RAM-only, job dir deleted on fetch).
KNOWN DECODE GAP (pre-existing, shared with faceswap; #1 follow-up): the `mp4`
0.14 demuxer parses the whole moov strictly and FAILS on a_girl.mp4's AAC audio
box ("mp4a box contains a box with a larger size than it") though we only need
the H.264 track. Audio-stripped clip decodes+swaps perfectly. Real AAC-LC phone
clips work (faceswap-validated). Improved the error to an actionable hint (codec.rs,
both decode sites). ROBUST FIX TODO: audio-tolerant demux (strip audio trak from
moov via box surgery, or a lenient demux crate) so arbitrary uploads decode.
JUDGMENT CALL for user: source FACE image is written to the job dir (plaintext,
deleted on fetch) like the existing input_image pattern; only the large VIDEO is
encrypted-at-rest per the explicit ask. Offer: make the face RAM-only too.

QUALITY EYEBALL: PASS (2026-07-05). 320x448-area x13f x16step (464x304 out, 185s
incl cold 5.68GB fp32->bf16 transcode): a REAL, coherent, identity-preserving
swap -- source face (an_1.jpg) mapped onto the a_girl target (mic/podcast scene),
target pose/hands/lighting/wardrobe preserved, face well-integrated, TEMPORALLY
STABLE across frames (no flicker/drift). Stages all finite (denoised std 0.657,
rgb in [-1,1]). Frames scratch/dreamidv/eyeball/dreamidv_0NN.png. Model is
CORRECT (parity) AND good QUALITY. Speed = FastWan-class as expected (report the
real product number after footprint work lets 480p run). The 320x448 run hit a
transient VAE OOM that SELF-RECOVERED via the wan VAE budget-reseed path.

PERF/VRAM FOLLOW-UP (post-correctness, expected for the 8GB card): 480x832x17f
OOM'd on full Wan2.1 VAE decode (worked at smaller dims). DreamID currently runs
bf16 streamed blocks (WanVariant block_transcode None) + FULL Wan2.1 VAE. Levers
(same as other Wan models, all EXIST): (1) Q8_0 block_transcode (footprint +
DP4A-ish; parity-gate), (2) VAE spatial tiling (wan VAE already has the tiled/
budget-reseed path) or tiny-VAE role (taew2_1, wire like anyflow), (3) coopmat+
fast_sdpa (native, default-on lever). Product 480p needs (1)+(2). Report REAL
trace number after. The cold fp32->bf16 DiT transcode (5.68GB) per run is a
cold-start cost (per-request in serve); GPU Q8FromBf16 prep or a pre-transcoded
bf16/Q8 safetensors would remove it.

NEXT (exact, in order; wan-core is mine, serial):
1. DiT FORWARD SURGERY (dit.rs forward_with_taps, gated on cfg.ref_conv; THE
   crux; other Wan models must stay bit-identical): (a) 48ch patch input is
   pipeline-fed (grid uses in_channels 48 for patchify, out_channels 16 for
   unpatchify -- verify WanDitShape/PatchGrid split in vs out); (b) when ref_conv:
   patchify img_ref [16,1,h,w] spatially -> ref_conv linear -> [ref_rows=pph*ppw,
   inner] prefix; prepend into the residual buffer (alloc rows=ref_rows+n_tok,
   ref at offset 0, video patch-embed at offset ref_rows); rope over grid
   (ppf+1,pph,ppw); block loop over rows; head/proj_out+unpatchify on x[ref_rows..]
   only (strip). Add `img_ref: Option<&[f32]>` to WanDitInputs.
2. open_dreamidv_source (source.rs) + WanVariant::dreamid_v() (pipeline.rs,
   vae=wan2_1(), no moe, block_transcode per perf later) + manifest role set
   (DIT_DREAMIDV + VAE_WAN21 + tokenizer N/A since baked context). Land WITH a
   caller so no dead_code.
3. dreamidv pipeline (new module): VAE-encode video(norm0.5)/mask(no norm)/image
   (norm0.5, white-pad) -> denoise image-CFG (2 fwd/step, scale 4.0, UniPC shift5
   16 step) -> VAE-decode. Baked context load.
4. e2e parity gate (conformance krea-style): torch pyref of dreamidv_wan_faster at
   tiny dims on committed a_girl fixtures; component (VAE-enc, ref_conv tokens, 1
   block) then full. Commit tiny fixtures.
5. app wiring (VideoModelId::DreamIdV + executor run + request video+source-image
   inputs) + encrypted video upload (base64->job dir, vault derive_cipher/seal/
   unseal made pub for at-rest) + face-swap web exposure + CLI/web parity.
6. serve redeploy (ASK before stopping the running server).

## PENDING USER EYEBALL: User LoRA vault (BUILT + e2e-VERIFIED + DEPLOYED, uncommitted)

Krea 2 Turbo is DONE (ported, perf-tuned, app-wired; see the resolved sections
below). The USER LORA VAULT feature is BUILT across every interface, e2e-proven
with a synthetic adapter, and the new serve is redeployed on :8080. No quality
shortcuts, no fp8, no baked adapter. Do NOT commit. Only open item: a real
Civitai-adapter eyeball by the user (see REMAINING below).

LANDED 2026-07-04 (uncommitted; compiles clean, clippy-clean, fmt'd; vault +
fold unit tests green; serve + cli binaries link):
- `thinfer_app::vault` (NOT a new crate -- user vetoed; native `vault` feature
  both cli+serve enable). Argon2id KDF (per-vault salt) + AES-256-GCM. Encrypted
  index grouped BY MODEL id; each entry seals name+extra (`enc_meta`); content
  blobs are random-id files holding raw ciphertext (no base64 bloat). Verifier
  ciphertext gates every op -> wrong password = one opaque `VaultError::Auth`
  (no oracle). STATELESS: key re-derived per op, plaintext dropped. DISK reveals
  only blob count/sizes, never which adapters/for which model. Ops: list/add/
  open/remove + `download(url, token)` + `ensure_safetensors` (validates by
  CONTENT, not extension -- Civitai URLs are extensionless) + `resolve_dir`
  (explicit > `THINFER_VAULT_DIR` > `<hf-cache>/vault`, so CLI+serve share one
  vault by default).
- Generic fold PROMOTED from `ltx::lora` to `thinfer_models::common::lora`
  (auto-discovery, stacking, both key conventions, rank-per-tensor, shape/enc
  preserving). `ltx::lora` is now a re-export -> LTX/Wan shipped paths untouched.
- Executor: `ImageRequest` gains `lora: Vec<LoraRef>` + redacting `Secret`
  password + `vault_dir`. `run_krea` decrypts vault entries -> in-memory
  `BytesOpener` safetensors source -> `common::lora` fold wrapping the DiT ->
  generic `krea_generate`. 0-site guard. Password never logged (Secret Debug =
  `<redacted>`). Gated on `ImageModelId::supports_adapters` (Krea2 only for now;
  fold is model-agnostic so a new image DiT opts in + wires its path).
- serve: `POST /vault/adapters/{list,add,remove}` (blocking crypto on
  spawn_blocking; add = download->ensure_safetensors->encrypt), `ImageSpec` gains
  `lora`+`password`, `ServeConfig.vault_dir`, OpenAPI-documented, auth-gated.
- CLI: `thinfer vault {add,list,remove}` + `generate image --lora NAME_OR_ID
  [:WEIGHT]` (local resolve against the vault; --remote adapter use errors).
  Password via hidden `rpassword` prompt or `THINFER_VAULT_PASSWORD` (never a
  flag). `--vault-dir` override.
- web: per-model Adapters as one COLLAPSIBLE `<details>` section (spans the form
  width, own internal grid -- kept out of the label/input grid so it doesn't
  wrap-mess), shown only for adapter models; password field (memory-only),
  List/Add-by-URL(+token)/Remove, multi-select checkboxes + per-adapter weight;
  `lora`+`password` on the image job spec.

VERIFIED e2e 2026-07-04 (GPU handed over): full path proven with a SYNTHETIC
scratch adapter (random weights on 28 `blocks.N.attn.wq` sites, rank 8, F32;
never committed). `thinfer vault add` via a localhost URL -> download +
ensure_safetensors (56 tensors) -> AES-GCM store (random-id 11MB blob + 496B
index; grep confirmed NO plaintext name on disk) -> `vault list` decrypts name/
size/weight. Then `generate image --model krea-2-turbo --seed 42 --lora
synth-style` vs the same base run: images DIFFER (mean-abs 50.9/255, 93.8% of
pixels changed) and BOTH valid (std 52.8 -> 60.2, no NaN). Confirms
decrypt->fold->generate applies. Base 512x512x8 ~119s, adapter run ~110s.
SERVE REDEPLOYED over HTTPS: `serve.toml` (tls_self_signed=true) -> self-signed
cert (SANs localhost/127.0.0.1/LAN IP), listening https :8080 (RTX 5070, coopmat
on); plain HTTP now refused. Boot logs the vault dir (`<hf-cache>/vault`), UI
ships the collapsible Adapter section, POST /vault/adapters/list -> 200 {[]} on
the empty default vault, openapi documents all three vault paths. Local serve
ALWAYS runs self-signed HTTPS now (secure context for WebCrypto + vault password
off plaintext LAN); launch with `--config serve.toml`.

CARRY-FORWARD GOTCHA: on Windows a running serve LOCKS
`target/release/thinfer-serve.exe`, so `cargo build --release` silently leaves
the old exe ("Access is denied (os error 5)" -- cargo still exits 0 for the rest
of the workspace). STOP serve before a release rebuild, or the "redeployed"
server is stale. (First redeploy this session hit exactly this; the CLI binary
updated fine since nothing held it.)

LANDED 2026-07-05 (uncommitted; fmt+clippy clean; 6 vault + 2 rename tests green):
- REAL-ADAPTER KEY FIX. The "Krea 2 Official Loras" (Civitai 2726235) fold 0
  sites: they ship DIFFUSERS names (`transformer.transformer_blocks.N.attn.to_q`,
  `img_in`, `final_layer`, `ff.gate`, `time_embed`, `txt_in`, `time_mod_proj`)
  while the base DiT is sd.cpp `krea2` GGUF (`blocks.N.attn.wq`, `first`,
  `last.linear`, `mlp.gate`, `tmlp`, `tproj.1`, `txtmlp`). NEW `krea/lora.rs`
  `lora_key_renames()` = diffusers->sd.cpp map (264 sites x2 halves = 528 keys,
  exactly matching the real file's tensor count), swaps `transformer.`->
  `diffusion_model.` + preserves `.lora_{A,B}.weight`. executor `run_krea` fold
  path wraps each adapter in `RenamedSource::with_passthrough(raw, renames)`
  before `discover_specs` (same pattern as the encoder's qwen3_gguf_renames).
  Verified by pulling the real safetensors header off Civitai and diffing every
  key vs loader.rs. DEPLOYED to serve :8080 (rebuilt+restarted 2026-07-05).
- VAULT = MULTI-PASSWORD now (user-requested). Dropped the vault-wide verifier:
  each entry sealed independently under a key from ITS add-time password; `list`
  skips entries it can't decrypt (wrong-pw list = `[]`, no oracle); `add` accepts
  any password (no "one true password"); `open`/`remove` fail closed with Auth;
  `remove` authorizes by decrypting the target. Shared salt => list still does 1
  Argon2 (no perf hit) + backward-compatible (serde ignores old verifier fields,
  existing entry stays readable under its original pw). FOOTGUN: a typo at add
  silently seals under the typo'd key. NOT yet redeployed (serve still runs the
  pre-vault-change binary; needs a rebuild+restart -- ASK first).

REMAINING (user, at leisure): eyeball a REAL Civitai krea2 adapter end to end
via the web UI's Add-by-URL -- now UNBLOCKED by the rename fix (should fold 264
sites). Only a real adapter proves real-key compatibility + aesthetic effect.

## USER LORA VAULT (design reference; full "how" in plan-details.md)

Decisions resolved with the user 2026-07-04 (all IMPLEMENTED, see NOW above):
per-model scoping (adapters belong to a model); NOT a new crate (module in
thinfer-app, native `vault` feature); vault dir defaulted+overridable, shared by
CLI+serve; Civitai token = separate transient field; password gates ADAPTER use
only (base models stay open); stack multiple adapters (Vec fold, per-adapter
weight); safetensors validated BY CONTENT not extension (Civitai URLs are
extensionless). Firm invariant held: disk access must not reveal WHICH adapters
you hold. Engine-side crypto/fold design lives in plan-details.md ("User LoRA
vault") so it is not duplicated here.

--- Krea 2 Turbo arch reference (below) ---

Facts (from web research 2026-07-04, confirm vs sources before coding):
- Arch: 12.9B diffusion transformer, novel (layerwise + refiner text-fusion
  blocks; NOT FLUX-based). Reference impl = leejet/stable-diffusion.cpp
  (docs/krea2.md + krea2 source) -- READ IT for exact structure.
- VAE: **Wan2.1 VAE** (z16, 8x/4x) -- REUSE `wan::vae` (WanVaeConfig, Wan2.1
  latent stats). Already cached (Wan-AI/Wan2.2-T2V-A14B-Diffusers).
- Text encoder: **Qwen3-VL 4B**, taps 12 hidden layers via a contrastive
  txtfusion projector (positive mid ~L14, negative deep L23/26/29/32). Base to
  adapt = `z_image::text_encoder` (Qwen3-4B, same backbone). Qwen3-VL-4B GGUF
  already cached (unsloth/Qwen3-VL-4B-Instruct-GGUF).
- Scheduler: turbo 8-step (confirm 8 vs 12), flow-matching CFG-off. Reuse
  `qwen_image::build_steps` (FlowMatchEuler + dynamic mu shift) as starting pt.

Template: `qwen_image` (GGUF image DiT + Qwen-VL tap + Wan-VAE reuse +
FlowMatchEuler CFG-free t2i). New dir `thinfer-models/src/krea/`.

Weights to fetch (exact repos pending arch-agent): base Krea 2 Turbo DiT GGUF
Q8_0 (canary) + Q4_K_M (default) from molbal/krea2-gguf or
realrebelai/KREA-2_GGUFs; Qwen3-VL-4B encoder (cached); Wan2.1 VAE (cached).

ARCH CONFIRMED from `third-party/stable-diffusion.cpp/src/model/diffusion/
krea2.hpp` (authoritative) + diffusers Krea2 config. Krea2 = **single-stream
MMDiT**, [txt ++ img] concat through all blocks:
- DiT: features 6144, layers 28, heads 48 / kv_heads 12 (GQA), head_dim 128,
  patch 2, in/out latent ch 16 (packed 64), SwiGLU inner
  ceil((2*6144/3)*4,128)=16384, norm_eps 1e-5. RoPE 3-axis {32,48,48} theta
  1000, Flux-style joint ids (txt at origin, img at (0,h,w)).
- Block (KreaSingleStreamBlock): shared 6-way adaLN from timestep (tproj ->
  [6*features]) PLUS per-block learned offset (mod.lin [6*features]); attn has
  per-head (1+w) QK-RMSNorm + a sigmoid OUTPUT GATE (out *= sigmoid(gate(x)));
  SwiGLU MLP. KreaRMSNorm = rms_norm then *(1+scale) [gemma-style bake].
- Text path: txtfusion = 2 layerwise blocks (attn over the 12-layer axis per
  token, dim 2560, 20 MHA heads, SwiGLU x4) -> projector Linear(12->1) collapse
  -> 2 refiner blocks (attn over tokens) -> txtmlp: RMSNorm ->
  Linear(2560->6144) -> gelu(tanh) -> Linear(6144->6144). (NO contrastive
  pos/neg projector; that web claim was FALSE.)
- Timestep: sinusoid dim 256 (max_period 10000, scale 1000) -> tmlp
  Linear(256->6144) gelu Linear(6144->6144); tproj gelu -> Linear(6144->6*6144).
- first: Linear(64->6144). last: KreaRMSNorm + 2-way final adaLN (from the
  pre-tproj t embedding) + Linear(6144->64).
- Encoder: Qwen3-VL-4B, tap 12 hidden states at layer idx {2,5,8,11,14,17,20,
  23,26,29,32,35} (every 3rd of 36; idx0=embeds). Prompt template system-prefix
  start_idx 34, suffix 5 tok, max_seq 512. Base to adapt = z_image Qwen3-4B
  encoder (same 2560/36/32-8/hd128 backbone) + capture 12 taps not just [-2].
- Scheduler: FlowMatchEuler, Turbo fixed mu=1.15, 8 steps, guidance 0 (CFG off).
  sigmas linspace(1,1/N,N)++0 then shift. (Confirm vs shipped scheduler_config.)
- VAE: Wan2.1 (z16 8x/4x) = REUSE wan::vae; cached at Wan-AI/Wan2.2-T2V-A14B-
  Diffusers vae/diffusion_pytorch_model.safetensors (diffusers keys). Need
  Wan2.1 latent mean/std (not Qwen-Image's).

WEIGHTS (cached unless noted): DiT GGUF realrebelai/KREA-2_GGUFs
TURBO/Krea-2-Turbo-{Q8_0 canary 13.6G, Q4_K_M default 7.2G} (DOWNLOADING);
encoder unsloth/Qwen3-VL-4B-Instruct-GGUF Q5_K_M cached (grab Q8_0 for canary);
VAE cached. GGUF TENSOR NAMES: inspect the downloaded GGUF before writing
loader (sd.cpp ggml names != on-disk gguf keys; likely diffusers/comfy naming).

BUILD ORDER:
1. Inspect Krea GGUF tensor names (gguf reader) -> loader weight-id map.
2. Scaffold `krea/`: mod(config), vae(WanVaeConfig+Wan2.1 stats), scheduler,
   packing/rope (crib qwen_image 3-axis but Flux-joint ids), text_encoder
   (Qwen3-VL 12-tap), dit (single-stream + txtfusion), loader, manifest, pipeline.
3. t2i e2e parity gate: pyref dequants the SAME Q8 GGUF; small dims; PNG stage.
   Doubles as perf rollup (gpu_ms by pipeline). Component parity first
   (encoder tap, txtfusion, one DiT block, vae) then full e2e.
4. Wire model-id + executor + CLI/serve/web parity + conformance feature.
5. serve redeploy (base krea in the web model list).

DEFERRED CHECKS: max_image_seq_len 4096 vs 6400 (only matters for Raw's
computed mu; Turbo uses fixed mu=1.15 so N/A); confirm Qwen3-VL-4B LM dims ==
Qwen3-4B via gguf meta before wiring encoder.

GGUF WEIGHT MAP CONFIRMED (realrebelai TURBO Q4_K_M, arch="krea2", 430 tensors,
sd.cpp-native names; weights Q4_K, norms/biases/mod F32, embedders F16). All
linears BIAS-FREE except {first, tmlp.0, tmlp.2, tproj.1, txtmlp.1, txtmlp.3,
last.linear}. GGUF ne = [in, out].
- Top: first.{weight[64,6144]F16,bias}; tmlp.0[256,6144]+b, tmlp.2[6144,6144]+b;
  tproj.1[6144,36864]+b; txtmlp.0.scale[2560], txtmlp.1[2560,6144]+b,
  txtmlp.3[6144,6144]+b; last.norm.scale[6144], last.linear[6144,64]+b,
  last.modulation.lin[6144,2]F16.
- txtfusion.projector.weight[12] (Linear 12->1, no bias, = 12-vec).
- txtfusion.{layerwise,refiner}_blocks.{0,1}: prenorm/postnorm.scale[2560],
  attn.{wq,wk,wv,wo,gate}[2560,2560], attn.qknorm.{q,k}norm.scale[128],
  mlp.{gate,up}[2560,6912], mlp.down[6912,2560]. (text MHA 20x128, SwiGLU 6912.)
- blocks.0..27: prenorm/postnorm.scale[6144], mod.lin[36864],
  attn.{wq,wo,gate}[6144,6144], attn.{wk,wv}[6144,1536] (GQA 12x128),
  attn.qknorm.{q,k}norm.scale[128], mlp.{gate,up}[6144,16384], mlp.down[16384,6144].
- KreaRMSNorm bake: effective scale = (stored_scale + 1) [gemma-style], applies
  to every *.scale above (prenorm/postnorm/qknorm/txtmlp.0/last.norm).

LANDED (uncommitted, compiles + 7 krea unit tests green): krea/{mod(config),
vae(Wan2.1 reuse),scheduler(FlowMatch turbo mu=1.15 N=8),loader(full weight
map),packing(Flux channel-major c/ph/pw = qwen layout),rope(Flux-joint ids: txt
at origin, img at (0,r,c) uncentered, axes{32,48,48} theta1000, interleaved-pair
-> op_rope)}; registered in lib.rs. Both DiT GGUFs (Q8_0+Q4_K_M) downloaded.
Reuse decisions: packing/rope reuse RopeEmbedder+op_rope (NOT half-rot); VAE =
wan::vae::wan2_1(); PNG = z_image::pipeline::encode_png.

NEXT (crux): krea/dit.rs. Mirror qwen_image/dit.rs pattern (custom forward on
lower ops + a KreaDitPipelines wrapping common BlockPipelines + extra pipelines).
Common Block driver is UNUSABLE (asserts hq==hkv; Krea is GQA 48/12 separate
wq/wk/wv + sigmoid gate + gemma norm + shared-mod).

OP INVENTORY (all exist unless noted): op_rmsnorm, op_sdpa(hq,hkv GQA-capable),
op_rope (interleaved-pair), op_silu_mul, op_add, matmul via common/embedders
linear_no_bias/linear_bias; core ops GeluF32, MulF32, TanhF32, BcastModulateF32
(x*(1+scale)+shift), BcastFmaF32 (x*s+y gate-residual), BcastMulF32. qwen_image/
dit.rs shows op_modulate(BcastModulate), mod_signal(chunk k of the 6-way mod
proj), biased matmul, GeluF32 dispatch. **NEW OP NEEDED: SigmoidMulF32 (out =
a*sigmoid(b), no 2x) for the attn output gate -- add in thinfer-core/src/ops
(template silu_mul.rs/tanh.rs; wire GeluF32-style pipeline in KreaDitPipelines).
gated_head_mul does NOT fit (per-head + 2x).**

(1+w) GEMMA NORM: KreaRMSNorm = rms_norm(x) * (stored_scale + 1). op_rmsnorm
does rms_norm(x)*w. Handle via op_bcast_affine on the SCALE at load OR a norm
variant OR simplest: op_rmsnorm with w then... cleanest = pre-add 1.0 to scale
on GPU once (weight_prep) OR apply (1+w) in a fused norm. DECIDE in dit.rs;
lowest-risk = op_rmsnorm(x, ones)+... no -- just do rms then bcast_fma(normed,
scale, normed) = normed*scale+normed = normed*(1+scale). One extra op/norm.

DiT FORWARD (per krea2.hpp Krea2Model::forward):
- img = pack -> first Linear(64->6144)+bias.
- t = sinusoid(timestep, dim256, period1e4, scale1e3); temb = tmlp0->gelu->tmlp2
  (each +bias) -> [6144]. tvec = gelu(temb) -> tproj Linear(6144->36864)+bias.
- txt: reshape encoder taps [txt_tok, 12, 2560]; txtfusion.layerwise[0,1]
  (attn over the 12 axis, full MHA 20x128, gated, gemma norm, SwiGLU6912) ->
  permute -> projector Linear(12->1) collapse -> [txt_tok,2560] ->
  refiner[0,1] (attn over tokens). Then txtmlp: rmsnorm(txtmlp.0.scale) ->
  Linear(2560->6144)+b -> gelu(tanh) -> Linear(6144->6144)+b.
- hidden = concat[txt(6144-wide) ++ img], rope freqs over [txt++img].
- 28x block: mods = chunk6(tvec + mod.lin);
  a = modulate(prenorm(x), scale=mods[1], shift=mods[0]);
  attn_out = gated GQA attn(a, rope) [wq/wk/wv, qk (1+w)norm, sdpa, *sigmoid(
  gate(a)), wo];  x += attn_out * mods[2] (bcast_fma gate-residual);
  m = modulate(postnorm(x), scale=mods[4], shift=mods[3]);
  x += SwiGLU(m) * mods[5].
  NB attn q/k/v projected from the MODULATED input `a`; gate(a) too.
- last: mods2 = chunk2(last.modulation.lin[2,6144] + t_embed[6144]);
  x = modulate(last.norm(x), scale=mods2[1], shift=mods2[0]) -> last.linear
  (6144->64)+b; slice img rows; unpatchify.
MODULATE ORDER CONFIRMED (flux.hpp:412): modulate(x, shift, scale) computes
x*(1+scale)+shift. krea2 block calls modulate(x, mods[1], mods[0]) so the 6-way
chunk = [0]scale_msa [1]shift_msa [2]gate_msa [3]scale_mlp [4]shift_mlp
[5]gate_mlp. Final layer 2-way = [0]scale [1]shift. (BcastModulateF32 wants
scale+shift; pass accordingly.)

USER ASK 2026-07-04: report 512x512 gen speed once measurable (after e2e gate
runs). 512px -> 64x64 latent -> 32x32=1024 img tok + ~txt; 8 steps Q4_K_M ~7.2GB
stream/step; likely streaming-bound. Give the REAL trace number, no guess.

LANDED (compiles clean): thinfer-core SigmoidMulF32 op (ops/sigmoid_mul.rs,
registered ops/mod.rs) = fused out=a*sigmoid(b), 3 storage variants, for the
attn gate. NO standalone conformance case yet (harness golden is in-rust, not a
keyed py ref -> extend risk); the e2e parity gate is its semantic proof. Op sigs
CONFIRMED: op_sdpa(q,k,v,mask,dst,b,s_q,s_k,h_q,h_kv,head_dim,scale,has_mask) =
native GQA (h_q!=h_kv ok); op_sdpa_f16 = mixed-prec fast path (PERF lever, use
for the DiT self-attn where q/k/v are post-norm/rope); op_rope(src,freqs,dst,
rows,heads,head_dim) interleaved-pair; matmul via common/embedders linear_no_bias
/linear_bias; op_bcast_fma(x,s,y)=x*s+y for gate-residual; BcastModulateF32 via
qwen op_modulate. Krea DiT act dtype = BF16 (residual > f16 range, like qwen).

LANDED (compiles clean, clippy-clean, fmt'd): **krea/dit.rs** -- the full DiT.
KreaDitPipelines (BlockPipelines + gelu + sigmoid_mul + rmsnorm_gemma);
block_cfgs(quant) bf16 acts; matmul Site routing (Qkv/Proj/FfnUp/FfnDown/Adaln);
gated_attention (GQA + (1+w) qk-norm + optional rope + sigmoid gate + wo),
swiglu, block_forward (shared-mod chunk6 + 2 gate-residuals), text_block_forward
(txtfusion block, no mod/rope), prepare_txt (layerwise x2 -> host projector
contraction -> refiner x2 -> txtmlp, ONCE per gen), forward (per-step: first
img embed + timestep tmlp/tproj + concat[txt++img] + 28 blocks + final mod +
slice img). Added 2 core ops: SigmoidMulF32, RmsNormGemmaF32 (both compile).
Modulate order per spec. Velocity via download_act readback. DEFAULT_QUANT=Q8_0.
PERF TODO (post-parity): fast_sdpa f16 path (op_sdpa_f16), i8 DP4A matmul sites,
depth-2 block submit -- all deferred until parity green.

ENCODER (next): Qwen3-VL-4B text backbone CONFIRMED == z_image Qwen3-4B (36L,
2560, 32/8 GQA, hd128, ffn9728) EXCEPT rope theta 5e6 (Qwen3=1e6). qwen3vl mrope
collapses to 1D rope for text-only prompts. GGUF arch "qwen3vl", tensor names ==
z_image qwen3_gguf_renames (blk.N.attn_{q,k,v,output,q_norm,k_norm}, ffn_*,
token_embd), cached Q5_K_M. 12 taps = hidden_states[{2,5,8,11,14,17,20,23,26,29,
32,35}] = running residual after that many layers (max 35 -> run 35 layers,
capture at those counts, order = select_layers). Build krea/text_encoder.rs
reusing z_image Qwen3 block machinery w/ theta 5e6 + 12-tap capture + Krea chat
template (system prefix start_idx 34, suffix 5, max_seq 512). Output
[txt_tok,12,2560] -> DiT.prepare_txt.

LANDED (WHOLE krea engine-side model COMPILES + clippy-clean + fmt'd):
- krea/text_encoder.rs: KreaTextEncoder (Qwen3-VL-4B, theta 5e6, reuses z_image
  Qwen3Block/Qwen3Handles/register_qwen3_handles/embed_lookup; runs 35 layers,
  snapshots residual after {2,5,...,35} -> [txt_tok,12,2560]).
- krea/pipeline.rs: KreaPipeline (encode->prepare_txt(once)->FlowMatchEuler
  denoise->Wan2.1 VAE decode->RGB; encoder cfg per-site catalog probe for mixed
  K-quants; evict_all_and_free between phases; reuses z_image encode_png).
  load(backend,residency,max_seq,quant) + generate_rgb + generate.
All 9 krea modules compile: mod/config, vae, scheduler, loader, packing, rope,
dit, text_encoder, pipeline. + 2 core ops (sigmoid_mul, rmsnorm_gemma).

NEXT: krea/manifest.rs (roles: DIT_GGUF_Q8_0/Q4_K_M realrebelai/KREA-2_GGUFs
TURBO/*; ENCODER_GGUF unsloth/Qwen3-VL-4B-Instruct-GGUF; VAE Wan-AI/Wan2.2-T2V-
A14B-Diffusers vae; TOKENIZER Qwen/Qwen3-VL-4B-Instruct tokenizer.json) with
qwen3_gguf renames for the encoder GGUF. Then app wiring (ImageModelId::Krea2Turbo
+ ImageKind + executor run_krea + request + wire/api + web app.js). Then
conformance krea/e2e.rs (Q8 canary, small dims, pyref dequants same GGUF; gives
512x512 speed). Then serve deploy.

*** THAT 64x64x2 "MILESTONE" WAS A FALSE GREEN (corrected 2026-07-04). *** The
health test only asserts FINAL-RGB finiteness, and the Wan VAE launders all-NaN
latents into a finite flat tan (std 0.39 = the vignette, NOT an image). The
pipeline was emitting NaN from the very first op. HARDEN the health test: assert
intermediate finiteness (encoder taps + pre-VAE latent) -- instrumentation for
this now exists (diag events "krea encoder taps stats", "krea pre-VAE final
latent stats", per-block NaN scan; gate them into an assert).

TWO NaN ROOT CAUSES FOUND + FIXED 2026-07-04 (uncommitted):
(1) DiT top-level linears (first, tmlp_0/2, tproj, txtmlp_1/3) are Q8_0/Q4_K in
    the GGUF, NOT F16 -- but dit.rs routed them through Site::Adaln whose cfg was
    bf16 -> Q8_0 bytes read as bf16 = NaN. FIX: block_cfgs matmul_adaln = q
    (Quant), and matmul_site Adaln branch passes bp.dequant_adaln (infra already
    built it; just unwired). The author's "F16 embedders ride bf16 Adaln" comment
    was WRONG: only last.modulation.lin is F16 (and it is read raw, not matmul'd).
(2) Encoder mixed-K quant. Q5_K_M encoder GGUF bumps attn_v/ffn_down -> Q6_K (the
    standard llama.cpp _K_M recipe) while q/k stay Q5_K. The krea encoder shares
    ONE BlockPipelines across all 35 layers; DequantStep.scheme is fixed at
    compile from the q_proj probe (Q5_K) -> Q6_K v/ffn_down dequant with wrong
    scheme -> NaN at layer 0. FIX: manifest ENCODER_GGUF -> uniform Q8_0
    (Qwen3-VL-4B-Instruct-Q8_0.gguf, downloaded; verified all linears+token_embd
    uniform Q8_0). NB the shared-single-pipeline encoder can't do per-layer mixed
    K-quant (z_image DiT compiles per-(layer,slot) pipelines; the Qwen3 ENCODER
    machinery does not). A small-download default tier would need per-layer-slot
    encoder pipelines OR a K->Q8 upload transcode.

AFTER FIXES (512x512x8): finite everywhere, velocity has spatial structure
(vel_spatial_std 0.13), and the PNG is a REAL image -- a red apple on a table
(scratch/krea_512_png/krea_e2e.png). Semantic conditioning WORKS.

QUALITY RESOLVED 2026-07-04 (3rd bug fixed): the darkness + cross-hatch WEAVE was
last.modulation.lin. It is an F16 [2, DIM] raw scale/shift pair the final layer
reads DIRECTLY as two rows -- but the loader registered it via register_linear,
which applies TransposePolicy::Linear2D to F16, SCRAMBLING scale/shift per channel
-> systematic per-output-channel (=per-subpixel) bias = the weave, and wrong
overall level = the darkness. Localized diag-first: pre-VAE latent 2D-FFT peaked
at Nyquist (period-2 latent = period-1 patch = intra-2x2-patch alternation);
per-subpixel mean spread 0.447. FIX: new common/loader::register_raw_param (F16/
F32/bf16, TransposePolicy::None, narrows to bf16; register_passthrough rejects
F16, register_linear transposes) -> krea last_mod uses it. AFTER: subpixel spread
0.447->0.010, Nyquist peak gone, latent mean -0.22->-0.05, rgb std 0.30->0.46 max
0.81->1.0. 256x256x8 = a GORGEOUS photoreal red apple on a wooden table
(scratch/krea_256_png). Packing interleave was RULED OUT (c*4+ph*2+pw matches
sd.cpp patchify patch_last=true). CARRY-FORWARD GOTCHA: F16 params the model reads
RAW (not matmul'd) must use register_raw_param; register_linear's Linear2D
transpose silently scrambles them (no NaN, just wrong -- shows as structured
artifacts). Only last.modulation hit this (first/last.linear are matmul'd Q8_0).

PERF LANDED 2026-07-04 (coopmat + fast_sdpa, user-approved). Was COMPUTE-bound on
dense bf16 matmul (NOT streaming): matmul_ffn_up 82s, ffn_down 51s, proj 30s, sdpa
25s, qkv 23s. i8-DP4A is UNAVAILABLE here (f16-only path; krea's big DiT needs a
bf16 residual -- same reason qwen_image block matmuls stay dense). So the lever is
COOPMAT (tensor cores, f16-cast at the matmul boundary, bf16 residual preserved).
Wired all 4 block sites (had to ADD coopmat_ffn_up to common CoopmatSites +
BlockPipelines -- Wan only coopmats proj/ffn_down since it i8's qkv/ffn_up) +
fast_sdpa (op_sdpa_f16). Krea matmul_site now routes dispatch_matmul_site_coopmat.
RESULT 512x512x8: 265s -> 99s (UNDER 2 MIN). Rollup: ffn_up 82->24s, ffn_down
51->7.8s, proj 30->6.2s, qkv 23->4.7s, sdpa 25->2.5s (~4.7x compute). QUALITY GATE
(user rule = opt-out iff quality imperceptibly changed): 256x256x8 A/B coopmat vs
dense -- BOTH gorgeous photoreal apples, NO degradation (equal fidelity); latent
diverges (MAD 8.6%, corr 0.70) but that's chaotic 8-step compounding of f16
rounding = a seed nudge, not quality loss. => DEFAULT ON (opt-out). Kill switch
THINFER_KREA_NO_COOPMAT=1 (fast_sdpa stays on). NATIVE ONLY -- web must keep
coopmat off (desktop-Vulkan). PNGs: scratch/krea_{256c,512c}_png.

--- prior (now-corrected) milestone note kept for the run cmd ---
e2e health at 64x64x2 (debug): tokenize->encode->txtfusion->28-layer DiT->Wan2.1
VAE->RGB. Run cmd:
  THINFER_TRACE=1 THINFER_POWER_PREF=high THINFER_E2E_STEPS=2 THINFER_E2E_WIDTH=64
  THINFER_E2E_HEIGHT=64 THINFER_E2E_PNG_DIR=<scratch> RUST_LOG="info,thinfer::
  diag=warn" cargo test -p thinfer-conformance --features krea-e2e --test krea
  e2e_health -- --nocapture --test-threads=1
Bundle: DiT realrebelai Q8_0 (cached), enc unsloth Qwen3-VL-4B Q5_K_M (cached),
VAE Wan2.1 (cached), tokenizer Qwen/Qwen3-VL-4B-Instruct (downloaded this sess).

RUNTIME BUGS FIXED this session (all in the run above): (1) rmsnorm_gemma
compiled with block bf16-weight cfg -> panic; (2) passthrough F32 scales NARROW
TO BF16 on GPU upload -> gemma norm needs bf16-weight variant (added
WGSL_BF16_PACKED_WBF16, (1+w)); (3) projector read assumed F32 48B, actually
bf16 24B -> read tl*2 bytes + bf16->f32 parse. KEY LESSON: passthrough 1-D F32
weights (norm scales, projector) arrive on GPU as BF16 under bf16 acts.

RESUME HERE (fresh context, in priority order):
1. DONE 2026-07-04: quality verified (gorgeous apple), 3 parity bugs fixed
   (Adaln quant, encoder Q8, last_mod transpose) -- see the resolved sections
   above. All 6 CAUTION parity risks below turned out FINE (sin/cos, mrope,
   Wan denorm, txtfusion, K-quant [fixed via uniform Q8], start_idx).
2. DONE 2026-07-04: perf via coopmat+fast_sdpa (512x512x8 265s->99s, <2min,
   default ON opt-out). See PERF LANDED above. Q4_K_M default still open as a
   footprint option (separate download; not needed for the vault work).
3. DONE + VERIFIED 2026-07-04: APP WIRING. thinfer-app model.rs (Krea2Turbo +
   ImageKind::Krea2, all arms), executor.rs (run_krea + map_krea; union DiT +
   RenamedSource(enc, qwen3_gguf_renames) + VAE; tokenize_t2i; KreaPipeline::load
   4-arg + QuantKind::Q8_0), request.rs (Krea2 in required_files role arm;
   validate needs nothing -- t2i rejects input-image via the _ arm). serve
   api.rs/wire.rs/CLI/index.html = NO CHANGES (generic over ImageModelId). web
   app.js (MODELS.image + MODEL_STEPS krea-2-turbo:8). TOKENIZER: reused
   HfTokenizer + tokenize_t2i (no new code); DROP_IDX 34 handled engine-side.
   Defaults 1024x1024x8 (aesthetic native; 512 = <2min fast tier). VERIFIED: CLI
   `generate image --model krea-2-turbo` @512x512x8 = 93.6s, PNG PIXEL-IDENTICAL
   to the e2e coopmat image (proves app HfTokenizer == python pyref tokens).
4. DONE 2026-07-04: serve rebuilt + REDEPLOYED with base krea-2-turbo in the web
   model list (listening https :8080). Serve runs the native engine so coopmat is
   on; the "coopmat off on web" rule is only for the wasm substrate.
5. NEXT (build after /clear): the USER LORA VAULT feature -- see its own plan
   section below. No adapter is baked in the repo.

CAUTION for e2e (unverified parity risks to watch -- bisect if 512 img is bad): (1) timestep sinusoid
sin/cos order (used qwen cos++sin flip_sin_to_cos); (2) mrope->1D collapse for
Qwen3-VL text (theta 5e6); (3) Wan2.1 latent norm applies in Krea (denorm in VAE
decode); (4) txtfusion tap order + projector host-contraction; (5) mixed K-quant
per-site routing for encoder; (6) chat-template tokens (start_idx 34). All fall
out of the pyref e2e diff -- diag-first when it runs.

## PRIOR TRACK (paused 2026-07-04): AnyFlow perf

AnyFlow-Wan2.1-T2V-14B is PORTED + VALIDATED + SERVE-DEPLOYED (details in the
track section below). 2-step quality at 480p is GORGEOUS on simple prompts.
The problem is WALL TIME. **User's full serve run, 832x480x81f 2-step, seed
4450053076214346960 (`scratch/browser-run/{log.txt,video.mp4}`): encode 7.5s,
step 1 = 635.5s, step 2 = 609.8s, VAE 169s, total 1424s.**

**ROOT-CAUSED 2026-07-02 (phase telemetry in `forward_block_tiled`, diag
`tiled block phases` event): the 81f block is ~13.2s wall = SDPA 9.0s +
pass-B (FFN+cross) 2.7s + pass-A 1.0s + streaming ~0.9s (overlapped).
`sdpa_sg` at s=32760 full attention is ~22 TFLOP/block and runs at ~2.6
TFLOPS-eff: INSTRUCTION-bound (per-key subgroupShuffleXor reduce chain +
scalar softmax per Q row), NOT DRAM-bound (WG 128->256 halved K/V traffic,
gained only 5%) and NOT streaming-bound (all four old suspects were minor).
The user's "10s at 0% GPU" was Task Manager's 3D graph missing the compute
queue.**

LANDED (uncommitted, all clippy-clean; sdpa_sg conformance 13/13 GREEN):
- Cross-attn Q -> i8 site (`QkvSite::CrossQ` shares the self-attn i8
  pipelines; normed A-side). Was dense-pipeline, ~2.5s/block at 81f (~35% of
  33f DiT GPU). K/V from un-normed text stay dense (unchanged).
- sdpa_sg WG 128->256 (`SG_WG`, web-safe at the default 256 limit).
- Depth-2 submit pipeline in the tiled block loop (pass A/B tiles submit
  ahead one; transient reserve accounts 2 tiles). Neutral at 81f (SDPA
  dominates) but strictly better; keep.
- Web UI: attn-window row now visible for anyflow (blank = full attention).
- Per-block phase telemetry (diag-gated) in `forward_block_tiled`. Needs
  `RUST_LOG=info,thinfer::diag=debug` (THINFER_TRACE alone filters to info).
Result: 81f block 15.9s -> ~12.8s (~16% e2e). W=3 attn-window opt-in cuts
SDPA ~3x more (projected ~7.3s/block, ~750s total) -- USER EYEBALL gate
(multi-subject risk, HY second-cat lesson).

LANDED 2026-07-03 (uncommitted, clippy-clean):
- **Tiny VAE (taew2_1) for AnyFlow, VALIDATED**: new manifest role
  `TINY_VAE_WAN21` (`lightx2v/Autoencoders taew2_1.safetensors`, z16 patch-1,
  same `decoder.{1..22}` keys as lighttaew2_2 -- the shared taehv decoder
  needed ZERO graph changes); per-model tiny-role pick
  (`VideoModelId::wan_tiny_vae_role`) in executor + request; web UI vae
  dropdown un-pinned for anyflow. **CRITICAL latent-space fact: taew2_1
  consumes the DiT's NORMALIZED latents as-is (TAESD convention); only the
  lightx2v `lighttae*` retrains take the full-VAE `z*std+mean` denorm
  (upstream gates on "lighttae" in the filename -- lightx2v pipeline.py).**
  `WanVariant.tiny_vae_normed_latents` carries this; wrong transform =
  washed-gray output (tiny-vs-full frame MAD 0.19 pre-fix -> 0.016 post-fix
  at same latent). Env-gated e2e arm: `THINFER_E2E_TINY_VAE=1 anyflow_e2e`.
  Decode 0.2s vs full 4.0s at 256x256x9f; kills the 169s VAE at 81f.
- **Web bug fixed: anyflow request payload never sent `attnWindow`** (row was
  visible but the spec omitted it -> W=3 silently ran full attention). Also
  sends the vae choice now. NEEDS SERVE REBUILD+REDEPLOY before the user A/B.
- `THINFER_SDPA_SG_CL` env override (models/common block compile) for the
  CL=4 A/B; NOT bit-exact across CL (reduce order), gate like a kernel change.

**DEAD END 2026-07-03: i8 SDPA for wan full attention -- measured ~10x
SLOWER, do not enable.** At 832x480x33f (s=13,320) `sdpa_i8` = 16.6s/block
vs ~1.5s f16 sdpa_sg (O(s^2)-scaled from the 9.0s/81f baseline). The kernel
is thread-per-Q-row scalar unpack+fma (no dot4I8Packed, no subgroup co-op
over D); a DP4A rewrite of the QK dot fixes <=40% of per-key work (the
softmax-V accumulate can't dot4) and SDPA here is instruction-bound, so i8's
halved K/V bytes buy nothing. Only an sdpa_sg-shaped subgroup i8 kernel
could compete, and its only edge over f16 would be bandwidth we don't need.
The wiring stays (off by default): `THINFER_WAN_I8_SDPA=1` env opt-in in
wan/pipeline.rs; sdpa_i8 kernel gained Q-row chunking (`row0`) + dense
f16/bf16 out modes, conformance 7/7 GREEN incl chunk-bit-exactness;
`op_sdpa_impl` i8 branch now gates on paired inputs (dense callers fall
through). Parity note: i8 latent was statistically EQUIVALENT to the f16
baseline on FastWan video_e2e (pre-VAE slope 0.9866 vs 0.9835 baseline);
only the chaotic vae_rgb band missed (0.9275 vs 0.94 floor; baseline itself
passes by 0.0025). Numbers in scratch/{video_e2e_i8,video_e2e_base}.log.

**SDPA kernel-tuning verdicts (2026-07-03, all at 832x480x81f, s=32760,
vs the 9.0s/block CL=8/QR=1/WG256 baseline): the kernel is AT this card's
practical full-attention ceiling. Do not re-try:**
- CL=4 (2 shuffle hops, BR=64): sdpa 10.3s mean over 80 blocks = ~14%
  SLOWER (more per-lane serial D work loses to the saved hop).
- QR=2 (Q-register blocking, landed as `THINFER_SDPA_SG_QR`, default 1):
  sdpa ~9.4s = neutral-to-4%-slower; the doubled q/o register set
  (occupancy) ate the amortized K/V tile loads. Codegen + tests stay
  (bit-exact vs QR=1; sdpa_sg conformance 16/16 GREEN incl
  `sdpa_sg_qr2_bitexact_vs_qr1`); knob may help other adapters.
- i8 SDPA: see DEAD END above (~10x slower).
Remaining full-attention perf comes from changing the WORK, not the
kernel: attn-window W=3 (opt-in, ~3x SDPA) and the steps dial.

DONE 2026-07-03: **post-change 81f e2e GREEN** (832x480x81f 2-step, CL=4 +
tiny VAE, 1183s test: denoise 1169s incl cold transcode, tiny-VAE decode
6.6s (!) vs 169s full, latent std 1.058, video std 0.61, motion MAD 0.24,
frames GORGEOUS -- red-car clip eyeballed at product dims,
`scratch/anyflow_81f_cl4_png/`). At the deployed CL=8 kernel this implies
~1010s denoise -> **~1070s total for 5s 480p with vae=tiny (was 1424s)**.

SERVE REDEPLOYED 2026-07-03 (tiny-VAE + attnWindow/vae web fix live).
**USER EYEBALL 2026-07-03, 832x480x81f 2-step W=3 vae=tiny (browser, seed
10096709036188377068): PERF CONFIRMED -- encode 7.4s, steps 312.5s/299.7s,
tiny VAE 10.7s, total 632s (~10.5 min; was 1424s). QUALITY FAILED: scene +
wardrobe change every ~1-2s == the W=3 window horizon (21 latent frames,
+-3 visible ~= +-0.75s; frames beyond it share no attention path). Plain
all-steps W=3 on AnyFlow = DEAD for temporal identity.** Same mechanism as
the HY second-cat, expressed as drift.

**HYBRID STEP-WINDOWING PROVEN + SHIPPED 2026-07-03 (anyflow default).**
Engine A/B at 832x480x81f 2-step, same seed/prompt vs the full-attn red-car
reference: drift (MAD-vs-frame0 at frame 80) = 0.120 full / 0.216 W3-all /
**0.124 hybrid (step 0 full, W=3 from step 1)** -- hybrid RESTORES
full-attention temporal consistency; frame-80 eyeball scene-stable. Denoise
873s hybrid vs ~1030s full (CL8-normalized) vs 608s W3-all. Shipped as the
AnyFlow arm's default (`step_attn_window(.., default_from=1)` in
pipeline.rs): a user attn-window on anyflow now means "full step 0, then
windowed". Wan2.2 keeps its shipped all-steps W=3 (default_from 0) until
its own eyeball. `THINFER_WAN_WINDOW_FROM_STEP` env overrides both; e2e
takes `THINFER_E2E_ATTN_WINDOW`. No new user flag -> CLI/web parity holds
by construction. PNG evidence: `scratch/anyflow_81f_{cl4,w3all,hybrid}_png`.
**User tier map (5s 480p, tiny VAE): ~10.5 min W3-all (BROKEN quality),
~14 min hybrid W=3, ~17.5 min full attention.**

SERVE REDEPLOYED 2026-07-03 (2nd time): hybrid default + job-lifecycle log
lines (id/dims/seed/elapsed at info, NO prompt -- lost-tab seed recovery;
the engine's diag generate-start stays muted in serve).

LANDED 2026-07-03 PM (uncommitted, user directive: pick the fastest-at-quality
model and drive it down; user vetoed FastWan as primary = quality too low at
5B, so AnyFlow stays the target; FastWan wins banked anyway):
- **FastWan 5B Q8_0 block transcode** (was bf16: ~9.8GB re-streamed per step at
  480p = 3.5x streaming-bound vs its ~13s/step GPU compute; 81f 480p full-VAE
  decode also OOMs at 6GB -> tiny VAE is the product path there). Parity gate
  GREEN at 256 (vae_rgb slope 0.9426 vs 0.94 floor, same band edge as the bf16
  baseline). LongLive split to its own `WanVariant::longlive_2_0_5b()` (keeps
  bf16; its parity bands pin bf16 today).
- **Per-block weight-prep hoist for the tiled Wan path** (`PreparedTileWeights`):
  dispatch_matmul_site re-dequanted each block weight PER ACTIVATION TILE
  (32x redundant at 81f: dequant_i8_qkv_self/ffn_up + dequant_f16_proj/
  ffn_down = ~40s GPU of the 873s hybrid denoise, and the depth-2 tile
  pipeline held two tiles' weight scratch in flight). Now dequanted once per
  block into per-forward persistent buffers, filled alongside cross-K/V.
  Plus act-quant-once for the shared q/k/v A-side (hunyuan qkv_a_side
  pattern). GATED: forced-tiling (THINFER_DIT_TILE_ROWS=64) PNG BYTE-EXACT vs
  pre-change baselines, fastwan(i8) + anyflow(coopmat/f16) legs, twice
  (hoist, then +act-quant-once). MEASURED at 81f hybrid: blocks 872->845.5s
  (pass_b 220->202s, pass_a 70->61s, fill overhead +2s). SDPA now 553s = 65%
  of blocks; the ceiling verdict stands (CL/QR/WG A/Bs triangulate
  issue-bound ALU; packed-f16 math is unsafe for QK products - do not
  reopen). anyflow_e2e gained THINFER_E2E_PROMPT for quality eyeballs.
- **Cold-start note: worklog NEXT item 4 is ALREADY DONE** - residency's
  `prep_op` GPU-transcodes Bf16->Q8_0 whole-tensor at stream-in
  (`WeightPrep::Q8_0FromBf16`, visible as q8_0_from_bf16 in the rollup);
  no CPU cold transcode remains on the anyflow path.
- Phase-split fact (81f hybrid telemetry + per-pipeline gpu_ms): SDPA 541s
  (62%, at kernel ceiling), pass-B 220s (coopmat_ffn_down 73s + coopmat_proj
  50s + i8_ffn_up 33s + dequants 40s), pass-A ~70s. i8 DP4A runs 11-12
  TFLOPS-eff; coopmat 5-5.5 (see dead-end below: that IS its ceiling).
  NB the rollup's sdpa_sg line under-reports (~13s total = impossible);
  trust the W3-vs-full e2e delta (~2.6-2.8 TFLOPS-eff) for SDPA.

**HYBRID EYEBALL DONE 2026-07-03 (agent-eyeballed, 81f 832x480 2-step, same
seed/prompt A/B, drift-prone tracking-shot prompt): hybrid W=3 holds
composition/scene (no W3-all-style cuts) but shows SLOW ATTRIBUTE DRIFT over
5s (hat band appears, sunglasses appear, polka-dot scale grows, dog morphs
breed, pavement dissolves to sand). FULL ATTENTION same seed: identity SOLID
across all 81 frames. The drift is the WINDOW's, not the model's. NB the
pixel-MAD drift metric called them equal (0.124 vs 0.120) -- it is blind to
semantic attribute drift; eyeball remains the deciding gate. Tier map stands:
hybrid ~14 min = stable-scene/soft-identity, full ~17 min = solid identity.
Frames: scratch/anyflow_81f_eyeball{,_full}_png.**

LANDED 2026-07-03 sprint (uncommitted, agents + gates; all clippy-clean):
- **AnyFlow/Wan tiled**: deferred per-block setup (fill_mod+cross_kv+prepared
  fill in ONE submit_deferred scope, SDPA fence chains into pass B: 3 hard
  syncs/block removed, upper bound ~10s at 81f; the double-buffered
  overlap variant was analyzed and DROPPED: submit-interleave hazard +
  ~466MB reserve growth). **Tiled cross-attn K/V step-cache**
  (WanCrossKvCache reused; per-expert caches on MoE; byte replay after step
  0). Both scheduling/replay-only.
- **Hunyuan Q8 mod linears: WIRED BUT DISABLED (gate red, then re-greened
  dense)**: with the adaln Q8 wiring ON, dit_parity (default AND tiled) dies
  in a wgpu device panic (wgpu_core.rs:2253 binding/validation, not a band
  miss) while the tapless t2v_e2e passes -> suspect the taps/diag path or
  the adaln dequant binding. The plumbing (Site::Mod, dequant_adaln,
  reg_lin_q8) is committed but the two flip points are OFF (cfgs note in
  hunyuan_block_cfgs + registration in DitH::register; they MUST flip
  together). All hunyuan gates GREEN with it off. Fix + re-enable = next HY
  session's first item; the prize below stands:
  (worklog attack item 2): img_mod/txt_mod
  -> Q8_0 dequant-once at the adaln site (new Site::Mod; NO i8 acts, weight
  rounding only; ON under the default i8 config, THINFER_HY_I8=0 bisects).
  T2V forward streams 2.37GiB less; AR chunk forwards 1.19GiB less (~10% of
  the 13.5GB stream; ~30GiB less PCIe per 77f run). bf16 lightx2v takes the
  GPU stream-in transcode; F32 minWM stays CPU (known follow-up).
- **Wan full-VAE OOM fix**: temporal streaming already existed (diffusers
  feat_cache form, exact); the bug was the spatial tile sizer seeding at
  100% of budget with no margin/retry. Now: 0.82 seed safety + strict
  budget + balanced re-seed on OOM (LTX pattern) + config-aware calibration
  (Wan2.1 was over-reserving ~2.7x -> bigger tiles on the 169s path).
  THINFER_VAE_MEM=1 telemetry.
- **LTX** (audit agent): act-quant-once (-40% act_quant, -2112 disp/run),
  step-invariant uploads hoisted out of the denoise loop (~45MB/step PCIe),
  sst_out host-cached (2 blocking readbacks/step gone). Confirmed NO
  per-tile weight re-prep bug in LTX; cross-attn K/V legitimately per-step
  (sigma-modulated). Remaining LTX levers (NOT taken, need gates/sign-off):
  dense matmul_qkv sites = 54.9s of ~105s DiT GPU; prompt-encode phase 61s
  of 178s wall (streaming-bound, per-layer prefetch already optimal); VAE
  OOM-retile heuristic.
- **Wan2.2 MoE audit** (no code): LoRA fold = lazy per-tensor, compute-once
  cache, NOT repeated across steps (verified vs step1-step2 delta = 62.5s
  cold fold per expert); fold cache = ~15GB RAM per expert UNACCOUNTED
  (~30GB both, held to request end). Designed-not-implemented: CPU-side
  fold-warm of the low expert during high steps (~60s off step 3) +
  fold-cache clear at the expert switch (frees ~15GB mid-request).

NEXT (in order):
1. USER: pick the anyflow default story given the eyeball verdict above
   (hybrid default is quality-honest for scenes without a tracked subject;
   full attention is the identity tier). Then the 4-step full-attention
   cats/dog composition A/B (decides the quality-ceiling tier).
2. If the anyflow hybrid eyeball is good, run the same A/B for Wan2.2's
   default W=3 (its all-steps window has the same horizon mechanism).
3. Artifact-retention TTL option (delete-on-fetch cost a clip to a closed
   tab 2026-07-03): offered to user, no decision yet.
4. GPU bf16->Q8 transcode prep kernel exists already (`Q8_0FromBf16`); NB
   measured: umT5 + cold transcode + first-block prep = ~12 min wall in
   the e2e harness (serve first-touch overlap is much cheaper, ~26s
   step-1-vs-2 delta) -- localize only if cold-start matters.
5. Consider default vae=tiny for anyflow in web UI (CLI already defaults
   Tiny) -- 81f tiny eyeball GREEN engine-side (red-car frames + user clip
   decode); full stays the parity path.

**Pin-plan lesson (3 device losses 2026-07-02, reverted): do NOT pin a
weight slice when the working set >> VRAM budget.** Pinning starves the
same-size recycle scan -> arbiter reclaim during in-flight compute ->
deferred-destroy zombies -> physical VRAM overshoots the accounted budget ->
device lost (at 4.2G/3.2G/2.1G pins alike, 8GB card). The fully-streamed
steady state (same-size recycle, zero frees) is the stable regime. Z-Image
keeps its pin plan (whole DiT fits). See the NO-pin comment in
`wan/pipeline.rs::denoise_with`.

**Quality note (2-step, hard prompt):** rendering superb, COMPOSITION wrong:
two cats, pianist cat has human hands + sneakered feet, dog limbs smear in
motion (`scratch/browser-run/video.mp4`). Decisive A/B queued: SAME seed +
prompt at 4 steps -- if composition heals, steps is the user's quality dial
(already exposed); if not, it's the model's multi-subject ceiling. Rides the
post-deploy user browser run.

## HY15 TI2V: PARKED (user decision made 2026-07-02)

User picked ANCHOR SYNTHESIS for the TI2V fast-T2V path: text request ->
in-engine t2i makes frame 0 (Z-Image turbo; derive image prompt from the
motion prompt, rewriter machinery is the natural place) -> run the model in
its trained I2V regime. RENAME the model id honestly (it is I2V, optionally
fronted by t2i -- user: "don't call it ti2v"). Prereq: one I2V 77f pixel
validation (latent trend says mild decelerating drift, needs eyeball).
Do AFTER the AnyFlow perf push. TI2V perf items (Q8 mod linears, prefix
pinning) fold into the same engine work above. Drift root-cause record is in
the section below; do not reopen it.

## AnyFlow-Wan2.1-T2V-14B: SHIPPED reference (ported 2026-07-02)

Any-step flow-map distill; steps USER-CONFIGURABLE (CLI+serve+web, done).
LOW-RAM pyref policy honored (component pyref; block-streaming full-DiT pyref
is a follow-up).
FACTS ESTABLISHED (from third-party/AnyFlow + diffusers main
transformer_anyflow.py + HF configs, all verified):
- Arch = stock Wan2.1-14B (= `WanDitConfig::wan22_14b()` exactly; Wan2.1 VAE,
  umT5-xxl, diffusers tensor names -> NO rename layer needed). 1099 tensors.
- ONLY delta: `condition_embedder.delta_embedder.linear_{1,2}` (a second
  TimestepEmbedding fed sincos(r)); rt_emb = 0.75*temb(t) + 0.25*delta(r)
  (gate_value=0.25 config const, deltatime_type="r"); timestep_proj =
  time_proj(silu(rt_emb)); rt_emb also feeds final-layer modulation. Blend =
  BcastMul x2 + add (rows=1).
- Scheduler: linspace(1,0,steps+1), sigma-shift 5 (same formula as ours),
  x -= (sigma_t - sigma_r)*v == our standard Euler; ONLY new input: r =
  next sigma*1000 passed to the DiT per forward. guidance 1 (no CFG). Demo:
  480x832x81f, 4 steps, fps 16, seed via torch Generator.
- Weights: transformer/ 3 bf16 shards IN HF CACHE (27GB). umT5+VAE reuse
  the existing FastWan bundle roles (Wan2.1 VAE in/out=16).
- License NSCLv1 (noncommercial) - user informed 2026-07-02.
IMPLEMENTED (2026-07-02, all compiling clean, uncommitted): WanDitConfig
`delta_embedder` flag + `anyflow_t2v_14b()`; ConditionEmbedder optional
delta MLP + gated blend (GATE_VALUE 0.25, `gated_blend` via BcastMul/Add);
WanDitInputs.r_timestep (None on all legacy paths); AnyFlowSampler
(scheduler.rs, unit-tested incl 2-step grid [1000, 833.3]); pipeline AnyFlow
arm gated on cfg.delta_embedder (steps from params.steps, taps for parity);
WanVariant::anyflow_t2v_14b (Q8_0 block_transcode, bf16 acts, Wan2.1 VAE);
manifest roles DIT_ANYFLOW_1..3 + variant "anyflow-t2v-14b"; VideoModelId::
AnyflowT2v14b + executor arm + defaults (832x480, 16fps, wan22 cell cap);
web UI entry (steps visible, uncapped; vae pinned full). `--steps` already
existed CLI+wire (UniPC's), now feeds AnyFlow too. Conformance:
`tests/wan/anyflow_e2e.rs` (embedder parity vs
`gen_anyflow_embedder_ref.py` fp32 pyref -- low-RAM: reads only
condition_embedder tensors -- + engine health at 256x256x9f 2-step + PNGs).
VALIDATED 2026-07-02: `anyflow_e2e` GREEN (embedder parity slope 1.0008
rel_rmse 2.3e-4 vs fp32 pyref; 256x256x9f 2-step health 70s incl transcode).
832x480x33f 2-step eyeball: GORGEOUS (sharp coherent prompt-faithful car
clip; denoise 375s incl cold bf16->Q8 transcode, VAE 67s;
`scratch/anyflow_480p_png/`). One maiden-run bug found+fixed engine-wide:
dtype probe read the SOURCE encoding while registration transcoded blocks ->
Q8/bf16 pipeline mismatch -> NaN; `load_variant` now derives dit_w from
`variant.block_transcode` first. Serve REDEPLOYED with AnyFlow (steps field
live in web UI). FastWan video_e2e full-parity regression GREEN 2026-07-02
("parity OK vs pyref", all bands) -- legacy Wan models proven untouched.
Follow-ups: block-streaming full-DiT pyref; 81f long-clip quality verdict
(user run in flight at /clear time).

## HY15 TI2V drift: ROOT-CAUSED + CLOSED (2026-07-02, reference record)

Controls at 832x480 (all same seed/prompt, 45f/3-chunk latent-trend harness,
`THINFER_E2E_SKIP_DECODE=1 THINFER_AR_DIAG=1`):
- text-only shipping: latent std/chunk 1.256 -> 1.413 -> 1.580 (+12%/chunk)
- text-only PURE BF16: 1.259 -> 1.417 -> 1.592 (IDENTICAL -> quant innocent;
  f16 SDPA also exonerated by kernel read: all-f32 accumulators)
- I2V (frame-0 anchor): 1.145 -> 1.259 -> 1.335 (+10%, +6%, DECELERATING ->
  anchor damps; commit-KV telemetry: cache K flat, cache V inflates at deep
  blocks, block53 v_std +13% by chunk 2)
Mechanism: unanchored text-only AR rollout at product dims (OOD mode for the
i2v-trained dmd ckpt; 33 txt tokens vs 6240 img tokens/chunk can't anchor).
448x256 is clean (img:txt ratio 3.5x smaller). **Recache-T mitigation DEAD
(2026-07-02): `THINFER_HY_RECACHE_T` (env, plumbed in ar.rs) at 250 gives
1.479/1.653, at 15 gives 1.441/1.620 -- BOTH WORSE than baseline 1.413/1.580.
Upstream's stabilization knob amplifies, never damps, this drift.** No engine
bug: quant/f16-SDPA/faithfulness all exonerated. Remaining product options
(USER DECIDES at wrap-up): (a) ship with guidance (text-only good to ~45f at
480p, I2V for longer), (b) validate I2V 77f pixels (trend says mild) and make
I2V the long-clip recommendation, (c) anchor-synthesis chain (image model
makes frame 0 -> I2V) as a new feature.
Drift metric harness: `scratch/drift_stats.py` (pixels) + in-log per-chunk
latent/commit-KV stats. Perf note: img-only acquire measured ~2.5x on chunk
denoise (23min/5chunks@480p). OOM guard: 6GB VRAM budget OOMs at chunk-5 KV
alloc at 77f/480p; 5GB works.

**>>> BUG FIRST (before perf): progressive AR drift.** User's 77f (5-chunk)
browser run: chunk 1 GORGEOUS (dancing dog, better than the old T2V eyeballs),
then saturation/contrast blows up chunk-over-chunk into chaos by chunks 4-5
(`scratch/crazy_frames/`). **The quant-noise theory is DEAD: the SHIPPING
config (i8+coopmat+f16 SDPA) at 448x256x77f T2V-probe is CLEAN and
statistically identical to the pure-bf16 run (contrast 0.13->0.23, clip
1.5%->14% in BOTH; `scratch/drift_ship_png/` vs `drift_bf16_png/`,
`scratch/drift_stats.py` is the metric). The earlier bisect lacked a positive
control; everything is clean at tiny dims.** Drift is REGIME-dependent:
resolution/KV-length (tokens/frame 390 -> 1560, s_k to ~32k rows where f16
SDPA + bf16 KV cache errors have 4x the accumulation length) and/or
content/prompt. IN FLIGHT: shipping-config repro at 832x480x77f T2V-probe
(duck prompt). If it reproduces -> bisect AT PRODUCT DIMS (i8-only vs
coopmat-only vs no-fast-sdpa). If clean -> content/prompt-dependent; add a
prompt override to the health test and retry with a motion-heavy prompt.
Mild clip%/contrast growth (to ~14% by chunk 5) exists even in pure bf16 and
is visually harmless; the discriminator is growth RATE (crazy frames hit 62%
clip). Upstream `stabilization_level` is 1 in every minWM config (= recache
t=0, what we do); >1 is a mitigation lever, not a faithfulness gap.

**It is currently WEIGHT-STREAMING BOUND, not compute bound: ~11min for 29f
(2 chunks) at 480p. Each of the `chunks*(4+1)` AR forwards re-streams all 54
blocks (~13.5GB mixed Q8/bf16) for only 6,240 rows of compute; the T2V's single
32,760-row forwards hide the same stream, chunk forwards can't. Attack in this
order:**

1. **DONE (uncommitted): skip txt-side weights in chunk forwards**
   (`acquire_img_block` in ar.rs; numerics-neutral). Perf A/B rides the
   in-flight 480p run vs the old 694s/2-chunk measurement.
2. **Q8 the modulation linears.** img_mod/txt_mod are `[12288, 2048]` = ~2.7B
   of the 8.3B params, currently dense bf16 (Module site). Dequant-once or
   direct-quant path cuts ~2.7GB/forward more.
3. **Pin a deterministic block prefix resident; stream only the tail.**
   Sequential LRU over a working set bigger than budget = worst-case thrash;
   the arbiter reclaimer (now registered) makes residency safe, but a fixed
   pin-split avoids re-uploading the SAME evicted blocks 25x.
4. **GPU F32->Q8_0 transcode prep kernel.** The 33GB F32 source transcodes on
   CPU per request (per-request isolation, no disk cache) -- minutes of cold
   start. A NarrowQ8 prep kernel (like `NarrowTransposeF32`) removes it.
5. **i8 KV cache** (store + upload i8 instead of bf16): halves the ~14GB host
   KV + its per-forward PCIe. Quality-gate (K/V are post-norm/rope, f16-safe;
   i8 unproven).
6. Gated deviation lever (later): skip the recache forward by caching the last
   denoise step's K/V (-20% forwards, trained-behavior deviation -> eyeball gate).

**T2V (non-causal) perf plan, still open:** perf-harness `gpu_ms`-by-pipeline
rollup on ONE DiT step to nail the SDPA-vs-matmul split, then i8/DP4A SDPA
(`cfgs.i8_sdpa` exists, unproven; gate on dit_parity bands) and DiT step/block
caching (gate). All ship as opt-in user options unless imperceptible.
Windowing stays opt-in (W=3 broke multi-subject coherence; default reverted to
full attention 2026-07-01). Rewrite perf (after DiT): condense the ~5.8k-token
system prompt (option + gate) and causal-aware prefill SDPA (quality-neutral,
default on). Reality check: seconds-at-quality is a HARDWARE ceiling on the
28-SM 8GB mobile 5070; kernel work buys ~2-3x, not 100x.

**Research watch:** no 1-2 step HY1.5 T2V distill exists (CF++ 1/2-step = Wan
2.1-1.3B only, below the quality bar -- user decision; minWM HY15 line is
4-step). Watch thu-ml/Causal-Forcing + MIN-Lab; the AR machinery here is
exactly what a few-step causal HY15 checkpoint would need.

## Lessons / dead-ends (do not retry)

- **Coopmat shared-memory staging is a DEAD END (measured 2026-07-03):** a
  multi-subgroup staged GEMM (ns 2x2..4x2, bk 16-64, llama.cpp shape) ran
  3.2-5.0 TFLOPS vs the direct kernel's 6.3-7.2 at the production site shapes;
  coopLoad from workgroup memory is strictly slower than L2-served global
  coopLoad on naga 29 + NVIDIA. Register-tile sweeps are flat (1x1 ~= 2x2 ~=
  4x4) -> the direct kernel is coopLoad-issue-bound, not reuse-bound; ~6-7
  TFLOPS is naga-coopmat's practical ceiling on this card (i8 DP4A does 11-12).
  Code reverted; only this note remains.

- **Weight transcode must catch every float source encoding.**
  `register_linear_transcode`'s Q8 arm matched Bf16 ONLY; F32/F16 fell through
  to a DENSE registration while the site's quant pipeline read Q8 -> garbage
  (F32 minWM DiT; the T2V "fp16" file is actually BF16 so it never showed).
  Fixed. Diag: garbage q/k look CLEAN after qk-RMSNorm; probe V/proj outputs.
- **Budgets: stream in/out under pressure, never predict reserves** (user,
  firm). Register `residency.reclaimer()` at `RECLAIM_EVICTABLE_WEIGHTS` on the
  arbiter so workspace growth evicts unpinned weights. The old Hunyuan
  carve-out reserves are gone; don't reintroduce.
- **Coopmat is a MATMUL-ONLY win.** Flash-attn coopmat SDPA measured 13x SLOWER
  than `sdpa_sg` (already ~12 TFLOPS-eff). cross-qkv coopmat DEVICE-LOSES
  (un-normed text > f16 65504). Coopmat also device-loses for M < WM(32); the
  `m >= wm` dense-fallback guard in `dispatch_matmul_site_coopmat` stays.
  naga gotcha: `var c: coop;` in a loop null-inits ONCE at fn entry.
- **DiT denoise is at the WGSL matmul ceiling** (latency/occupancy-bound, not
  bandwidth); bigger tiles BACKFIRE; weight-only quant does NOT speed compute.
  Measure via e2e `gpu_ms by pipeline`, NOT microbench.
- **Q8_0 is the quality+perf baseline for big DiTs** (Q4_K per-request fold was
  ~2x SLOWER and broke quality). Q4_K_M = footprint option only.
- **i8 DP4A**: qkv/ffn_up only (normed A-sides). proj/ffn_down carry outliers
  -> f16-cast coopmat OK, i8 acts NOT. Cross-attn-qkv from un-normed text: never.
- **BLOCK-WIDE f16 is a DEAD END** on big-DiT residuals (outlier channels >
  65504); bf16 residual is load-bearing. Mixed-precision f16 SDPA (post-norm
  Q/K/V cast only) is safe and shipped.
- **VAE decode is conv-GPU bound**; do NOT retry the conv3d im2col hoist
  (REVERTED, slower). Tiny (TAEHV) decoders are the fast default where they
  exist; LTX tiles its VAE (activation-bound peaks; seed BELOW budget, balanced
  re-seed on OOM).
- **Never run fp32 CPU pyref above tiny dims** (40GB host at 256x256).
- **LTX**: off-subject output = resolution/adherence regime, NOT text encoding
  (512+ two-stage widescreen is the regime). Gemma encoder MUST run F32 acts.
  The gemma `(1+w)` norm bake is REAL, don't "fix" it. Don't flip the LTX DiT
  to strict budget (relies on overshoot into device slack).

## Carry-forward gotchas (engine-general)

- **Ops reading aux params as f32 must dequant bf16 weights first** (f32
  binding fed bf16 bytes = silent garbage; conformance has no such dtype pair,
  so it passes conformance and fails in the model).
- **Q8_0 subnormal f16 scale bug is FIXED** (regression test exists); affects
  any tiny-weight quant tensor on the bf16-dequant path.
- **umT5 / large-residual encoders MUST run bf16 acts** (f16 overflows ->
  washed blob). Check non-finite, not just NaN.
- **Module-level matmuls need their OWN bf16 `matmul_module` pipeline**; never
  quantize a weight whose matmul site has no dequant step.
- **RoPE**: freqs pack to the act dtype; Wan RoPE3D = interleaved-pair, Qwen3 +
  LTX = half-rot. Hunyuan = interleaved-pair, img tokens only; AR chunks rope
  at ABSOLUTE frame positions.
- **GGUF tensor padding**: F32/F16 narrow arms slice `elements*size`, not
  `on_disk_bytes`.
- **Third-party clones**: `rm -rf <clone>/.claude` right after cloning.
- Video staging: per-frame PNG seq / contact sheet; MP4 in CLI only.

## Status (shipped -- DO NOT DISTURB)

- **hunyuan-video-1.5-ti2v** (causal AR, minWM dmd) -- ACTIVE, see
  `hy15-causal-plan.md`.
- **HunyuanVideo 1.5 T2V** (lightx2v 4-step, 480p) -- shipped; native 4B/8B
  prompt rewriter, tiny-ft VAE, joint windowed SDPA (opt-in), cancel wiring.
- **LTX-2.3 distilled + Sulphur-2** (22B joint AV) -- shipped, `ltx-plan.md`.
- **FastWan2.2-TI2V-5B** -- parity GREEN; UniPC default. PENDING user eyeball
  of a UniPC clip vs the KingNish Space.
- **LongLive-2.0-5B** (AR long/multi-shot) -- shipped. AR perf LANDED
  2026-07-03 (uncommitted): per-request cross-attn text K/V cache
  (`WanCrossKvCache`, byte-replay = bit-identical) + prefix segments written
  directly into the window buffers (kills ~39GB/chunk host gather + GPU-GPU
  copies; upload-once-per-chunk is INFEASIBLE: all-layer prefix K/V = 7.8GB >
  card). Gates: longlive_e2e + multishot GREEN at THINFER_LL_BUDGET_GB=6 (the
  8GB default OOMs on the 8GB card with today's desktop VRAM overhead).
  **DISCOVERED: longlive_parity vel_c0s0 FAILS PRE-EXISTING at HEAD
  (slope 0.9214 vs band 0.030, byte-identical with and without today's
  changes, so today's work is exonerated). The gate is stale vs weeks of
  kernel changes (f16 subgroup SDPA etc.); root-cause = bug-first item for
  the next LongLive session.** OPEN: re-measure WARM at 576.
- **Face-swap** -- shipped; NEXT = quality (XSeg + GFPGAN), `faceswap-plan.md`.
- **Ideogram-4**, **Z-Image**, **Qwen-Image(-Edit)-Rapid** -- shipped.
- **Wan2.2-T2V-A14B** -- shipped; attn-window default W=3 (long clips).
- **i8 DP4A matmul ON by default** (`--no-i8-matmul` = bf16 reference).

## Crate layout + serve (shipped reference)

- Layering: core < models (dep-clean) < native < app < {cli, serve}; wasm
  `thinfer-web` is its own substrate. serve = separate binary over shared
  `thinfer-app` (`JobExecutor` trait). Web UI is `include_str!`-baked -> web
  edits need a serve rebuild. One fresh wgpu device per JOB.
- **Deploy = I do it myself** (ASK before stopping the server he toys with):
  stop -> `cargo build -p thinfer-serve --release` -> `Start-Process` the exe
  with `-ArgumentList '--config','C:\work\personal\thinfer\scratch\serve.toml'`,
  `-WorkingDirectory <projects>`, stderr -> `scratch/serve.log`, stdout ->
  `scratch/serve.stdout.log`, `-WindowStyle Hidden`. Confirm "listening
  (https)" in the log. NEVER run the exe in a foreground Bash call. rustls is
  TLS-1.3-only; the browser is the check. serve.toml now sets `ram_budget=28G`
  (TI2V host KV cache ~14GB at 77f).
- DEFERRED: serve==CLI byte-parity test; disk-backed SSE ring buffer.

## Running the e2e / measuring

Card = RTX 5070 Laptop (8GB); keep budgets <8GB. All serial
(`--test-threads=1`). Always `THINFER_TRACE=1 THINFER_POWER_PREF=high` +
`THINFER_E2E_PNG_DIR` for staging; read the `gpu_ms by pipeline` rollup first.
- Causal TI2V: `cargo test -p thinfer-conformance --features hunyuan-e2e
  --release i2v_e2e_health -- --nocapture --test-threads=1`. Default 448x256x13
  (one chunk, minutes); `THINFER_E2E_{WIDTH,HEIGHT,FRAMES,VRAM_GB}` scale to
  product dims. `THINFER_I2V_T2V_PROBE=1` = text-only mode.
  `THINFER_AR_DIAG=1` = per-stage stats; `THINFER_HY_I8=0` bisects i8.
- Hunyuan T2V: `t2v_e2e` (pyref parity, tiny dims), `t2v_perf` (engine-only
  per-step bench, `--attn-window` sweep), `dit_parity` taps,
  `rewriter_caption_compare` (`--features qwen3-lm`, `--ignored`).
- FastWan: `video_e2e` (256x256, budget 6GB; `THINFER_E2E_SKIP_PYREF=1` for
  perf-only). LongLive: `longlive_parity`/`longlive_e2e` (256x256).
- LTX: `t2v_e2e_health` (`--features ltx-e2e`, 121f 512x320 6GB); component
  parity gates ONE AT A TIME (12.5GB encoder OOMs a multi-test binary).
- GGUF inspect: `uv run --with gguf python` + `gguf.GGUFReader`.
- CLI run: `THINFER_TRACE=1 THINFER_POWER_PREF=high thinfer generate ...
  --vram-budget 5G --ram-budget 5G` (TI2V needs bigger ram for the KV cache).
