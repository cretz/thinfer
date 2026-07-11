"""Qwen-Image VAE-encode parity reference (the edit path's reference-image
latent channel).

Loads the full `Qwen/Qwen-Image` `vae/` safetensors (the SAME file the engine
loads) into a diffusers `AutoencoderKLQwenImage`, makes a deterministic RGB
image `[3, H, W]` in `[-1, 1]`, encodes it, and dumps the raw distribution
parameters (mean ++ logvar, pre-normalization -- the pipeline normalizes
later):

  image.bin    f32 LE [3, H, W]          input image (CHW), the engine encodes it.
  moments.bin  f32 LE [2*z_dim, h, w]    encoder params (T=1 squeezed), mean
                                         (ch 0..z_dim) ++ logvar (ch z_dim..2z).
  meta.txt     "z_dim h_out w_out"

The engine encodes the SAME image with the shared Wan-family KL encoder; both
sides consume identical safetensors weights so the band is just the engine's
bf16-weight / f16-act rounding vs this fp32 reference. The edit path consumes
`latent_dist.mode()` = mean = channels 0..z_dim.

Memory: the VAE is ~0.2B; fp32 CPU at tiny dims (64x64) is trivial.

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image.gen_vae_encode_ref \\
        --vae <vae/diffusion_pytorch_model.safetensors> --out <dir> \\
        --width 64 --height 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from safetensors.torch import load_file

SPATIAL = 8
Z_DIM = 16


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assert args.width % SPATIAL == 0 and args.height % SPATIAL == 0, "dims must be /8"
    h_out = args.height // SPATIAL
    w_out = args.width // SPATIAL

    torch.set_grad_enabled(False)

    vae = AutoencoderKLQwenImage(
        base_dim=96,
        z_dim=Z_DIM,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
    ).eval()
    state = {k: v.to(torch.float32) for k, v in load_file(str(args.vae)).items()}
    missing, unexpected = vae.load_state_dict(state, strict=False)
    assert not missing, f"missing params: {sorted(missing)[:8]}"
    assert not unexpected, f"unexpected params: {sorted(unexpected)[:8]}"

    # Deterministic image [3, H, W] in [-1, 1] (CHW); encode wants [1, 3, 1, H, W].
    rng = np.random.default_rng(1)
    img = rng.uniform(-1.0, 1.0, (3, args.height, args.width)).astype("float32")
    _dump(torch.from_numpy(img), args.out / "image.bin")
    x = torch.from_numpy(img)[None, :, None]  # [1, 3, 1, H, W] (BCTHW)

    dist = vae.encode(x.to(torch.float32)).latent_dist
    moments = torch.cat([dist.mean, dist.logvar], dim=1)  # [1, 2z, 1, h, w]
    moments = moments[0, :, 0]  # [2z, h, w]
    _dump(moments, args.out / "moments.bin")
    (args.out / "meta.txt").write_text(f"{Z_DIM} {h_out} {w_out}\n")
    print(
        f"[py-qwen-vae-enc] done: z_dim={Z_DIM} latent={h_out}x{w_out} "
        f"moments{tuple(moments.shape)} "
        f"mean_range[{dist.mean.min():.3f},{dist.mean.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
