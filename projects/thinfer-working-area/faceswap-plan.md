# Face-swap plan (`thinfer generate face-swap`) -- SHIPPED

High-perf video face swap: input video + source image -> per-frame HyperSwap.
Self-contained from the video/LongLive work (DO NOT DISTURB those). Built on a
new generic ONNX executor (the reusable payoff).

CLI: `thinfer generate face-swap --input-video <mp4> --source-image <png|jpg>
--output <mp4> [--model hyperswap-1a|1b|1c] [--download-as-needed]`.
Validated end-to-end (Obama->Biden clip): decode -> swap (~0.154s/frame after
perf pass) -> encode mp4. Output visually correct.

## NEXT (ACTIVE 2026-06-22): correctness + resource fixes (precede QUALITY)

Found while validating face-swap on a real 4K phone clip
(`/c/work/personal/intabai/notes/test-inputs/`: `face.jpg` + `video.mp4` =
4096x2160, H.264 High profile, 2 B-frames, 224 frames @ 25fps; also
`video-big-audio.mp4`). All four issues DONE (UNCOMMITTED); re-validate
FastWan/LongLive mp4 output still byte-fine (shared encoder path touched) before
commit.

1. DONE: decode of High-profile / B-frame / 4K input. openh264 0.6->0.9 +
   `DecoderConfig::flush_after_decode(NoFlush)` + `flush_remaining()` drain (the
   crate's B-frame-safe pattern). 0.9 encoder API deltas in BOTH `video.rs` and
   `faceswap.rs`.

2. DONE: RAM/VRAM budget + STREAMING (was ~30GB: whole clip materialized twice).
   `faceswap.rs` rewritten to STREAM decode -> swap -> encode one frame at a time
   (`swap_video_streaming` + `Mp4VideoSink`), so peak host RAM is a few frames,
   not the clip. Added `--ram-budget`/`--vram-budget` (parse_budget) +
   `backend_for_stats`/`report_mem` `[mem]` rollup, matching video/image. NB: the
   ONNX executor has no residency pager (small resident models), so the budgets
   are informational/reporting here, not enforced -- the streaming is the real
   bound. VALIDATED on the 4K clip: ran end-to-end, 224 frames, no blowup.

3. DONE: encode resolution cap. `fit_encode_dims` downscales the swapped frame to
   fit openh264's encoder cap (<=3840 long edge / <=2160 short edge), aspect-
   preserved, even dims, applied per-frame in `Mp4VideoSink::push` before encode.
   VALIDATED: 4096x2160 -> 3840x2024, all 224 frames encode (was dying at frame 0).

4. DONE: audio passthrough. `extract_audio` reads the input AAC track and
   `mux_mp4` (extended with `Option<AudioPassthrough>`) remuxes it verbatim (no
   re-encode), timing preserved (sample start_time/duration in the audio
   timescale). GOTCHA: the `mp4` 0.14 crate models AAC as only
   `{profile,freq_index,chan_conf}` and DROPS the rest of the AudioSpecificConfig,
   so HE-AAC (SBR/PS) cannot round-trip -- it would emit a broken esds
   (96000Hz/0ch). So passthrough is gated to the object types that fully round-trip
   (Main/LC/SSR/LTP); HE-AAC etc. are DROPPED with a clear warning (no corrupt
   audio). VALIDATED: AAC-LC -> output 44100/2ch decodes clean; HE-AACv2 ->
   video-only + warning. (Real phone clips are AAC-LC.) To passthrough HE-AAC
   would need verbatim esds-byte copy, which the crate's high-level API can't do.

Perf seen on the 4K clip (healthy, matches prior): hyperswap ~138ms, detect ~16ms,
paste ~46ms, warp ~3ms per frame; streaming wall ~157s for 224 frames at 4K (paste
over the full 4096x2160 frame dominates; perf is the separate deferred item).
Run: `THINFER_TRACE=1 THINFER_FS_PROFILE=1 THINFER_POWER_PREF=high
./target/release/thinfer.exe generate face-swap --source-image <face.jpg>
--input-video <video.mp4> --output <out.mp4> --download-as-needed`.

## NEXT (deferred direction = QUALITY, not perf)

Current quality == intabai BASE path (SCRFD align -> HyperSwap 256 -> feathered
ellipse paste): good for clean frontal faces, but NOT best-reasonable. To match
intabai full quality, add (all are more ONNX models through the SAME executor +
the existing warp/paste pipeline -- the cheap part is done):
1. FACE ENHANCER (biggest perceptual win): GFPGAN 1.4 (facefusion/models-3.0.0
   `gfpgan_1.4.onnx`, ~340MB) / CodeFormer / RestoreFormer++. After swap, warp
   the swapped face to FFHQ-512 template (TEMPLATE_FFHQ_512, in pipeline.ts),
   run enhancer ((p/255-0.5)/0.5 RGB in, *0.5+0.5 out), paste back. CodeFormer
   takes a `weight` f64 input (0.7). See pipeline.ts `enhanceFace`. CLI flag
   `--enhancer gfpgan|codeformer|none` (default gfpgan?).
2. XSEG OCCLUSION MASK (facefusion/models-3.1.0 `xseg_1.onnx`, ~70MB, NEEDS the
   patch in intabai `patches/xseg_1.patch.json` -- a Max-op patch; check if our
   executor needs it or can run raw). Input 256 NHWC BGR /255. Output mask ->
   clip -> gaussian blur -> clip(0.5,1)*2-1 stretch (pipeline.ts getOcclusionMask).
   Combine with feathered via element-wise min in paste_back (cv.ts pasteBack
   occlusionMask path). Makes occluders (hands/hair/glasses) show through.
3. CHEAP: use HyperSwap's own `mask` output (2nd output, [1,1,256,256]) in the
   paste instead of ignoring it (min with feather). Already decoded, just unused.
NOTE: enhancer/xseg ONNX op coverage must be checked (run onnx_attrs.py on them);
likely the same conv-net op set the executor already handles. XSeg is NHWC (our
kernels assume NCHW) -- may need a transpose at the model boundary or an NHWC tap.

OBSERVED FAILURE MODES (eyeballed on the 4K clip, 2026-06-22): swap degrades hard
at (a) drastic head angle (strong yaw/pitch) and (b) face half off-screen / hands
crossing the face -- ghosting + double-edges round the jaw, identity stops holding.
ROOT CAUSE is the PIPELINE, not the swap net: (a) the rigid 5-pt SIMILARITY warp
cannot model out-of-plane rotation, so the 256 crop is off-distribution at extreme
pose; (b) the FIXED-ELLIPSE feather paste has no occlusion/segmentation awareness,
so it paints swapped pixels over hands/neck/background and misses the true face
boundary. Switching swap variant (1b/1c, same 256 arch) does NOT help these. The
fixes ARE the QUALITY items above, in this value order: XSeg/face-parse occlusion
mask (#2, the biggest fix for hands + off-screen bleed) > use HyperSwap's own mask
output (#3, free) > enhancer (#1, frontal sharpness) > pose-gating (down-weight or
skip the paste at extreme yaw). HyperSwap is a strong fast OPEN single-pass swapper
(>> old inswapper_128) but not the quality ceiling; the real ceiling is the post
stack (mask+enhancer), and beyond that, pose/occlusion-robust DIFFUSION swappers.
CANDIDATE TO EVALUATE LATER (user, 2026-06-22): DreamID-V
(github.com/bytedance/DreamID-V, weights hf.co/XuGuo699/DreamID-V) -- diffusion
video face-swap, likely much better pose/occlusion but 1-2 orders more compute/frame
(weigh against the thin-hardware goal + the per-frame budget below before adopting).

## What landed

GENERIC ONNX EXECUTOR (`thinfer-core/src/onnx/`, reusable beyond face-swap):
- `proto.rs` - hand-rolled protobuf wire reader (no dep), ONNX `ModelProto`
  subset; decodes f32/f16/i64/i32/f64 `raw_data` -> f32/i64.
- `shape.rs` - load-time shape inference + const-fold of the integer/shape
  subgraph (fixed input shapes => all static). Emits `Step::{View, Compute}`.
  Reshape/Flatten/Squeeze/Unsqueeze/Cast = metadata views; Shape/Gather/Concat/
  Slice/Unsqueeze/arith on consts folded to constants.
- `kernels.rs` - self-contained f32 WGSL: conv2d (groups+dilation+stride+pad),
  convtranspose2d, gemm, instance_norm, channel_affine (BN fold), prelu, unary
  (relu/sigmoid/tanh/leakyrelu), broadcast binary (add/sub/mul/div), expand,
  transpose (4D), depth_to_space (DCR/CRD), resize (nearest-asym + bilinear-
  halfpixel), maxpool.
- `exec.rs` - `OnnxModel::{load,run}`. Resident const buffers (incl. folded BN
  affine), one command buffer per forward over backend-allocated buffers, read
  outputs back. BatchNorm folded to per-channel affine at load.
- PARITY vs onnxruntime (gated `faceswap-e2e`, `tests/faceswap_onnx.rs`):
  SCRFD rel ~1e-6, ArcFace ~5e-6, HyperSwap ~1.2e-3 (fp16 weights -> f32). GREEN.

FACE-SWAP PIPELINE (`thinfer-models/src/faceswap/`):
- `image.rs` - Image (HWC RGB f32) + geometry ported from intabai cv.ts:
  similarity transform (LS solve), invert/warp affine (bilinear), feathered
  paste-back, bilinear resize. Unit tests committed (transform/invert/warp).
- `detect.rs` - SCRFD letterbox 640 + (p-127.5)/128 BGR + distance-decode
  (strides 8/16/32, 2 anchors) + NMS. Outputs grouped by shape (numeric names).
- `mod.rs` - `FaceSwapper::{load, source_embedding, swap_frame, detect}`.
  Templates arcface_112_v2 (embed) / arcface_128 (swap 256). ArcFace L2-norm.
  HyperSwap target (p/127.5-1), output (*0.5+0.5). Feathered paste (no XSeg).
- Pipeline e2e (gated, `tests/faceswap_pipeline.rs`): real photos, asserts
  faces detected at sane bbox, embedding unit-norm, swap NaN-free + localized
  (face region changes, corner untouched), writes PNG. GREEN.

CLI + VIDEO (`thinfer-cli/.../generate/faceswap.rs`):
- Model download from FaceFusion HF repos (deepghs/insightface buffalo_s,
  facefusion/models-3.0.0, models-3.3.0).
- Source decode via `image` crate (png/jpeg). Video decode: mp4 demux (`mp4`) +
  openh264 Decoder (AVCC->AnnexB, SPS/PPS prefix per AU). Output: openh264
  encode reusing video.rs mux helpers (now `pub(crate)`).

## Perf (measured 2026-06-21, RTX 5070 Laptop, 960x2182 1-face frame)

Per-frame 0.34s -> 0.154s (~2.2x); 48-frame clip 20.7s -> 12.2s. Breakdown now:
hyperswap GPU ~127ms (was 280), detect ~13ms, paste ~11ms, warp ~2.5ms.
Profilers (gated, kept): `THINFER_FS_PROFILE` (per-phase), `THINFER_ONNX_OPPROF`
(per-op-type GPU; absolutes inflated by per-submit latency, use for ranking).

LANDED (all quality-neutral, parity unchanged: hyperswap 1.2e-3, scrfd 1e-6):
- Tiled implicit-GEMM conv for group=1/dilation=1 convs (reuse `ops/conv2d`,
  3 tile regimes). The HyperSwap/ArcFace bulk. 280->157ms.
- ConvTranspose via zero-upsample + flipped/transposed-weight tuned conv
  (`ZERO_UPSAMPLE` kernel + load-time weight transform). 157->146ms.
- Expand elimination: AdaIN broadcasts are zero-cost views; the binary kernels
  broadcast from `[1,C,1,1]` directly. 146->127ms.
- Host warp/paste parallelized over rows (`std::thread::scope`). Marginal
  (paste is full-frame-scan + 25MB clone bound, not compute).

TRIED + REVERTED: bf16 conv weights. MEASURED SLOWER (127->139ms) + parity loss
(scrfd 6.4e-2). The HyperSwap convs are f32-FMA COMPUTE-bound on this GPU, not
weight-bandwidth-bound, so bf16 only adds unpack cost. DO NOT re-try bf16/f16
weight storage for speed (VRAM-only benefit). Same applies to f16 acts (conv
accumulates f32 regardless).

SPIKED + NOT WIRED: crop batching. Batch=8 through HyperSwap gives only 1.09x/crop
(parity bit-equal, 1.21e-3) -- the convs are NOT occupancy-bound at batch 1, so
batching does not pay. KEPT the batch-safe `CONCAT2` kernel (general 2-input
concat, replaces the batch-1-only contiguous copy; parity-verified) + the
`hyperswap_batch_spike` test (documents the 1.09x finding + guards batched
correctness). DO NOT build a batched pipeline -- measured dead.

PATH TO "a few minutes" (~3-5x more needed; not yet done):
- i8 DP4A conv (the repo's matmul 6x technique, applied to conv): the only real
  ALU lever for the compute-bound convs. Large + quality-sensitive build.
- Frame batching: deep-layer convs have tiny spatial (2x2/4x4) -> terrible
  batch=1 occupancy; running B face-crops together amortizes. Medium effort,
  helps occupancy-limited (not FLOP-limited) ops.
- Cheap: detect every Nth frame (~10ms/frame), bbox-only paste (~7ms/frame).

## Open follow-ups (low priority)
- Durable committed fixtures: e2e tests are env-gated on scratch images. A
  committed tiny face fixture + pyref would make them CI-runnable.
- XSeg occlusion mask + face enhancers (GFPGAN/CodeFormer) not ported (intabai
  has them; HyperSwap path works without).
- Multi-face: swaps every detected face with the same source (matches intabai).
- Temporal coherence (detect every Nth frame, smooth landmarks) not done.
- HE-AAC (SBR/PS) audio passthrough: dropped with a warning (crate can't round-trip
  its AudioSpecificConfig). AAC-LC/Main/SSR/LTP DO pass through verbatim (item 4 DONE).

## Pointers

Scratch: `scratch/faceswap/` - `onnx_inspect.py`/`onnx_attrs.py` (graph dumps),
`gen_golden.py` (onnxruntime goldens in `golden/`), `imgs/` (test photos+clip).
Reference: `intabai/web/src/video-face-swap/{pipeline,models,cv}.ts`.
