"""Qwen-Image-Edit host PREPROCESS parity reference (real PNG in).

The engine's edit path consumes host-prepared tensors (ViT patches, VAE image,
edit-templated tokens). `thinfer-app`'s `preprocess.rs` builds those in Rust;
this module builds the SAME tensors from a real PNG the way the checkpoint's
prescribed conditioning does: Phr00t's fixed ComfyUI
`TextEncodeQwenImageEditPlus` node (Qwen-Image-Edit-Rapid-AIO is an Edit-2509
merge), with the VL image then run through the `Qwen2VLImageProcessor` steps.

  * ViT channel: `common_upscale(.., "area")` to a 384^2 area (torch area
    interpolate, rounded sides), then smart_resize to a mult of 28, BICUBIC,
    CLIP normalize, merge-unit-major patchify -> `pixel_values [N, 1176]`.
  * VAE channel: an 896^2 area with sides floored to a mult of 32,
    `common_upscale(.., "lanczos", "center")` (center crop to that aspect, PIL
    LANCZOS), normalized to [-1, 1] -> `[3, 1, Hv, Wv]`.
  * Tokens: the node's Plus `llama_template` with `Picture 1: ` + the vision
    block (`<|image_pad|>` expanded to `n_img = (gh/2)*(gw/2)`) before the
    prompt; `drop` = the system + user-header tokens ComfyUI strips.

  pixel_values.bin f32 LE [N, 1176]
  vae_image.bin    f32 LE [3, 1, Hv, Wv] in [-1, 1]
  token_ids.bin    u32 LE
  meta.txt         "gh gw n_img image_pad_start Hv Wv drop"

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image.gen_preprocess_ref \\
        --image IN.png --processor-dir <dir-with-preprocessor_config.json> \\
        --tokenizer-dir <dir> --prompt "..." --out <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from thinfer_pytorch_ref.qwen_image.gen_encoder_edit_ref import IMAGE_PAD
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import MERGE

PLUS_SYSTEM = (
    "<|im_start|>system\nDescribe key details of the input image (including any objects, "
    "characters, poses, facial features, clothing, setting, textures and style), then explain "
    "how the user's text instruction should alter, modify or recreate the image. Generate a new "
    "image that meets the user's requirements, which can vary from a small change to a "
    "completely new image using inputs as a guide.<|im_end|>\n<|im_start|>user\n"
)
PLUS_TEMPLATE = (
    PLUS_SYSTEM
    + "Picture 1: <|vision_start|>{img}<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
)
VL_AREA = 384 * 384
VAE_TARGET_SIZE = 896

IMAGE_TOKEN_ID = 151655  # <|image_pad|>
PATCH = 14
TEMPORAL = 2
VIT_FACTOR = PATCH * MERGE  # 28
VIT_MIN_PIXELS = 56 * 56
VIT_MAX_PIXELS = 28 * 28 * 1280
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float64)


def _smart_resize(h: int, w: int) -> tuple[int, int]:
    """transformers `image_processing_qwen2_vl.smart_resize` (factor 28)."""
    f = VIT_FACTOR
    h_bar = round(h / f) * f
    w_bar = round(w / f) * f
    if h_bar * w_bar > VIT_MAX_PIXELS:
        beta = math.sqrt((h * w) / VIT_MAX_PIXELS)
        h_bar = max(f, math.floor(h / beta / f) * f)
        w_bar = max(f, math.floor(w / beta / f) * f)
    elif h_bar * w_bar < VIT_MIN_PIXELS:
        beta = math.sqrt(VIT_MIN_PIXELS / (h * w))
        h_bar = math.ceil(h * beta / f) * f
        w_bar = math.ceil(w * beta / f) * f
    return h_bar, w_bar


def _vit_pixels(img, gh: int, gw: int) -> np.ndarray:
    """The Qwen2VLImageProcessor pixel_values for a single image, torchvision-
    free: BICUBIC resize -> rescale 1/255 -> CLIP normalize -> merge-unit-major
    patchify (`permute(0,2,5,3,6,1,4,7)` + temporal repeat) -> [N, 1176]."""
    from PIL import Image as _Im

    hp, wp = gh * PATCH, gw * PATCH
    resized = img.resize((wp, hp), resample=_Im.BICUBIC)  # PIL takes (w, h)
    arr = np.asarray(resized, dtype=np.float64) / 255.0  # [H, W, 3]
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)[None]  # [1, C, H, W]
    patches = arr.reshape(1, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
    patches = np.repeat(patches[:, :, :, :, :, :, None], TEMPORAL, axis=6)
    flat = patches.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH)
    return np.ascontiguousarray(flat.astype("float32"))


def _area_resize(img, area: int):
    """The node's VL resize: scale to `area`, sides rounded, ComfyUI
    `common_upscale(.., "area")` = torch area interpolation."""
    import torch
    import torch.nn.functional as F
    from PIL import Image as _Im

    w, h = img.size
    s = math.sqrt(area / (w * h))
    nw, nh = round(w * s), round(h * s)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    out = F.interpolate(t, size=(nh, nw), mode="area")[0].permute(1, 2, 0).numpy()
    return _Im.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))


def _vae_resize(img):
    """The node's VAE resize: `target_size`^2 area, sides floored to /32, then
    `common_upscale(.., "lanczos", "center")` (center crop, PIL LANCZOS)."""
    from PIL import Image as _Im

    w, h = img.size
    s = math.sqrt(VAE_TARGET_SIZE * VAE_TARGET_SIZE / (w * h))
    vw, vh = int(w * s / 32) * 32, int(h * s / 32) * 32
    old, new = w / h, vw / vh
    cw, ch = w, h
    if old > new:
        cw = round(w * new / old)
    elif old < new:
        ch = round(h * old / new)
    x, y = (w - cw) // 2, (h - ch) // 2
    return img.crop((x, y, x + cw, y + ch)).resize((vw, vh), resample=_Im.LANCZOS), vh, vw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--processor-dir", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from transformers import AutoTokenizer

    img = Image.open(args.image).convert("RGB")
    in_w, in_h = img.size

    # --- ViT channel: the node's area resize, then Qwen2VLImageProcessor ---
    vl = _area_resize(img, VL_AREA)
    gh_px, gw_px = _smart_resize(vl.size[1], vl.size[0])
    gh, gw = gh_px // PATCH, gw_px // PATCH
    n = gh * gw
    pixel_values = _vit_pixels(vl, gh, gw)
    assert pixel_values.shape == (n, 1176), pixel_values.shape
    pixel_values.astype("<f4").tofile(str(args.out / "pixel_values.bin"))

    # --- VAE channel: the node's reference-latent image, in [-1, 1] ---
    resized, vae_h, vae_w = _vae_resize(img)
    vae_np = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    vae_np = vae_np.transpose(2, 0, 1).reshape(3, 1, vae_h, vae_w)  # CTHW (T=1)
    np.ascontiguousarray(vae_np).astype("<f4").tofile(str(args.out / "vae_image.bin"))

    # --- tokens: edit template, image_pad expanded to n_img ---
    mgh, mgw = gh // MERGE, gw // MERGE
    n_img = mgh * mgw
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    text = PLUS_TEMPLATE.format(img=IMAGE_PAD * n_img, prompt=args.prompt)
    drop = len(tokenizer(PLUS_SYSTEM, add_special_tokens=False)["input_ids"])
    enc = tokenizer(text, return_tensors="np", add_special_tokens=False)
    ids = enc["input_ids"][0].astype(np.int64)
    pad_positions = [i for i, t in enumerate(ids.tolist()) if t == IMAGE_TOKEN_ID]
    assert len(pad_positions) == n_img, f"{len(pad_positions)} pad != n_img {n_img}"
    pad_start = pad_positions[0]
    assert pad_positions == list(range(pad_start, pad_start + n_img)), "pad not contiguous"
    ids.astype("<u4").tofile(str(args.out / "token_ids.bin"))

    (args.out / "meta.txt").write_text(
        f"{gh} {gw} {n_img} {pad_start} {vae_h} {vae_w} {drop}\n"
    )
    print(
        f"[py-qwen-preprocess] in={in_w}x{in_h} ViT={gh}x{gw} N={n} n_img={n_img} "
        f"image_pad_start={pad_start} VAE={vae_h}x{vae_w} seq={ids.shape[0]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
