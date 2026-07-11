"""Qwen-Image-Edit pipeline-assembly HEALTH inputs.

The full-DiT byte-parity pyref OOMs (20B), so the e2e edit gate is a HEALTH
check, not numeric parity. This module produces the ENGINE INPUTS for that gate
from a single tiny deterministic test image, so the engine consumes correct
preprocessed tensors and the test isolates pipeline ASSEMBLY (vision -> edit
encode -> ref-latent -> [noise ++ ref] denoise -> decode) from preprocessing.

Two conditioning channels share the same source image at DIFFERENT resolutions
(exactly as `pipeline_qwenimage_edit.py` does: ViT smart-resizes to a mult of
28, the VAE to a mult of 32):

  * ViT channel: a `[N, 1176]` patchified tensor (HF processor merge-unit-major
    layout, via the shared `gen_vision_ref._patchify`), grid `gh x gw`. The edit
    chat template (drop_idx=64) is tokenized with `<|image_pad|>` expanded to
    `n_img = (gh/2)*(gw/2)` slots; we dump the token ids + the first image-pad
    index.
  * VAE channel: a `[3, 1, Hv, Wv]` image in `[-1, 1]` at a 32-multiple
    resolution (the edit ref-latent channel; `(z-mean)/std` normalize + pack is
    done engine-side).

Both images derive deterministically from the same RNG so the run is
reproducible. No model weights are loaded here (tokenizer-only), so this is fast.

  token_ids.bin    u32 LE, edit-templated prompt (image_pad expanded).
  pixel_values.bin f32 LE [N, 1176], ViT patch input.
  vae_image.bin    f32 LE [3, 1, Hv, Wv] in [-1, 1], VAE ref image.
  meta.txt         "gh gw n_img image_pad_start Hv Wv"

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image.gen_edit_inputs \\
        --tokenizer-dir <dir> --prompt "..." --out <dir> \\
        --gh 8 --gw 8 --vae-h 64 --vae-w 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thinfer_pytorch_ref.qwen_image.gen_encoder_edit_ref import EDIT_TEMPLATE, IMAGE_PAD
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import MERGE, _patchify

IMAGE_TOKEN_ID = 151655  # <|image_pad|>


def _vae_image(h: int, w: int, seed: int = 11) -> np.ndarray:
    """Deterministic [3, 1, H, W] image in [-1, 1] (CTHW, T=1)."""
    rng = np.random.default_rng(seed)
    img = rng.uniform(-1.0, 1.0, (3, 1, h, w)).astype("float32")
    return np.ascontiguousarray(img)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gh", type=int, default=8)
    p.add_argument("--gw", type=int, default=8)
    p.add_argument("--vae-h", type=int, default=64)
    p.add_argument("--vae-w", type=int, default=64)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.gh % MERGE == 0 and args.gw % MERGE == 0, "ViT grid must be /2"
    assert args.vae_h % 32 == 0 and args.vae_w % 32 == 0, "VAE dims must be /32"

    from transformers import AutoTokenizer

    mgh, mgw = args.gh // MERGE, args.gw // MERGE
    n_img = mgh * mgw

    # --- ViT channel: patchified pixels + edit-templated tokens ---
    pixel_values = _patchify(args.gh, args.gw)
    pixel_values.astype("<f4").tofile(str(args.out / "pixel_values.bin"))

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    text = EDIT_TEMPLATE.format(img=IMAGE_PAD * n_img, prompt=args.prompt)
    enc = tokenizer(text, return_tensors="np", add_special_tokens=False)
    ids = enc["input_ids"][0].astype(np.int64)
    seq = int(ids.shape[0])
    pad_positions = [i for i, t in enumerate(ids.tolist()) if t == IMAGE_TOKEN_ID]
    assert len(pad_positions) == n_img, f"{len(pad_positions)} image-pad != n_img {n_img}"
    pad_start = pad_positions[0]
    assert pad_positions == list(range(pad_start, pad_start + n_img)), "image-pad not contiguous"
    ids.astype("<u4").tofile(str(args.out / "token_ids.bin"))

    # --- VAE channel: ref image in [-1, 1] at a 32-multiple resolution ---
    vae_image = _vae_image(args.vae_h, args.vae_w)
    vae_image.astype("<f4").tofile(str(args.out / "vae_image.bin"))

    (args.out / "meta.txt").write_text(
        f"{args.gh} {args.gw} {n_img} {pad_start} {args.vae_h} {args.vae_w}\n"
    )
    print(
        f"[py-qwen-edit-inputs] seq={seq} image_pad_start={pad_start} n_img={n_img} "
        f"ViT={args.gh}x{args.gw} VAE={args.vae_h}x{args.vae_w}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
