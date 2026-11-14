"""Qwen-Image-2.1 VAE encode + decode parity reference (fp32, SAME safetensors).

Encodes a seeded smooth RGBA image (what the pipeline feeds for a condition
image: planar `[4, h, w]` in `[-1, 1]`) and decodes the normalized posterior
mean back, so the decode sees a realistic latent (random latents saturate the
output clamp).

  image.bin       f32 [4, h, w]        encoder input.
  mean.bin        f32 [64, h/16, w/16] posterior mean (raw, un-normalized).
  latent_norm.bin f32 [64, h/16, w/16] `(mean - latents_mean) / latents_std`.
  decoded.bin     f32 [4, h, w]        decode of `latent_norm` (denormalized here).

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image21.gen_vae_ref \\
        --vae-dir <snapshot>/vae --out <dir> --size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from thinfer_pytorch_ref.qwen_image21.gen_dit_ref import _dump, _locate_diffusers_src


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_diffusers_src())
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage21.from_pretrained(str(args.vae_dir), torch_dtype=torch.float32).eval()

    s = args.size
    # Smooth structure at a few scales; opaque alpha.
    field = sum(
        F.interpolate(torch.randn(1, 3, s // k, s // k), size=(s, s), mode="bicubic", align_corners=False)
        for k in (64, 16, 4)
    )
    rgb = torch.tanh(field[0] * 0.6)
    image = torch.cat([rgb, torch.ones(1, s, s)])
    _dump(image, args.out / "image.bin")

    mean = vae.encode(image[None, :, None]).latent_dist.mean[0, :, 0]
    _dump(mean, args.out / "mean.bin")
    lm = torch.tensor(vae.config.latents_mean).view(-1, 1, 1)
    ls = torch.tensor(vae.config.latents_std).view(-1, 1, 1)
    norm = (mean - lm) / ls
    _dump(norm, args.out / "latent_norm.bin")
    decoded = vae.decode((norm * ls + lm)[None, :, None]).sample[0, :, 0]
    _dump(decoded, args.out / "decoded.bin")
    print(
        f"[py-qwen21-vae] {s}x{s} latent{tuple(mean.shape)} decoded{tuple(decoded.shape)} "
        f"range[{decoded.min():.3f},{decoded.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
