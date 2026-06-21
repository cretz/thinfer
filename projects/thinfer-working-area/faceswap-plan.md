# Face-swap plan (`thinfer generate face-swap`) -- SHIPPED

High-perf video face swap: input video + source image -> per-frame HyperSwap.
Self-contained from the video/LongLive work (DO NOT DISTURB those). Built on a
new generic ONNX executor (the reusable payoff).

CLI: `thinfer generate face-swap --input-video <mp4> --source-image <png|jpg>
--output <mp4> [--model hyperswap-1a|1b|1c] [--download-as-needed]`.
Validated end-to-end (Obama->Biden clip): decode 48 frames -> swap (~0.34s/frame
at f32) -> encode mp4. Output visually correct.

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
- Audio passthrough from input video not done (video-only output).

## Pointers

Scratch: `scratch/faceswap/` - `onnx_inspect.py`/`onnx_attrs.py` (graph dumps),
`gen_golden.py` (onnxruntime goldens in `golden/`), `imgs/` (test photos+clip).
Reference: `intabai/web/src/video-face-swap/{pipeline,models,cv}.ts`.
