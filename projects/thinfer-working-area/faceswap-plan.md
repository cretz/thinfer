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

## Open follow-ups (low priority)

- PERF: convs are direct (1 thread/output elem), f32 weights. Wins available:
  bf16/f16 weight storage (halve VRAM, ~2x bandwidth), tiled implicit-GEMM conv
  (reuse ops/conv2d shape), batch frames, overlap decode/swap/encode. ~0.34s/
  frame now; fine but not optimized.
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
