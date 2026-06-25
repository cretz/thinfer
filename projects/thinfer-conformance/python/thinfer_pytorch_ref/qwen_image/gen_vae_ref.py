"""Qwen-Image VAE-decode parity reference.

Loads the full `Qwen/Qwen-Image` `vae/` safetensors (the SAME file the engine
loads) into a diffusers `AutoencoderKLQwenImage` (native diffusers tensor keys,
strict load), generates a deterministic NORMALIZED latent `[z_dim, 1, h, w]`,
applies the engine's `z * std + mean` per-channel denorm, decodes, and dumps:

  latent_norm.bin  f32 LE [z_dim, h, w]   normalized latent (pre-denorm, f=1).
  decoded.bin      f32 LE [3, H, W]       decoder RGB in [-1, 1].
  meta.txt         "z_dim h_in w_in"

The engine reads latent_norm.bin, applies the SAME denorm internally, and
decodes with the shared Wan-family KL-VAE; both sides consume identical
safetensors weights so the band is just the engine's bf16-weight / f16-act
rounding vs this fp32 reference (convs accumulate f32 in-kernel).

Memory: the VAE is ~0.2B; fp32 CPU at tiny dims (64x64) is trivial.

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image.gen_vae_ref \\
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

# vae_scale_factor(8) -> 8x spatial downscale; image => 1 latent frame.
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
    h_in = args.height // SPATIAL
    w_in = args.width // SPATIAL

    torch.set_grad_enabled(False)

    vae = AutoencoderKLQwenImage(
        base_dim=96,
        z_dim=Z_DIM,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        temperal_downsample=[False, True, True],
    ).eval()

    # Full diffusers checkpoint: native keys, bf16 on disk -> fp32 for reference.
    state = {k: v.to(torch.float32) for k, v in load_file(str(args.vae)).items()}
    missing, unexpected = vae.load_state_dict(state, strict=False)
    assert not missing, f"missing params: {sorted(missing)[:8]}"
    assert not unexpected, f"unexpected params: {sorted(unexpected)[:8]}"

    # Deterministic normalized latent [1, z_dim, 1, h_in, w_in] (BCTHW).
    rng = np.random.default_rng(0)
    latent_norm = rng.standard_normal((Z_DIM, 1, h_in, w_in)).astype("float32")
    _dump(torch.from_numpy(latent_norm), args.out / "latent_norm.bin")

    # Engine denorm: z = latent_norm * std + mean, per channel.
    mean = np.asarray(vae.config.latents_mean, dtype="float32").reshape(Z_DIM, 1, 1, 1)
    std = np.asarray(vae.config.latents_std, dtype="float32").reshape(Z_DIM, 1, 1, 1)
    z = torch.from_numpy(latent_norm * std + mean).unsqueeze(0)  # [1, z, 1, h, w]

    decoded = vae.decode(z.to(torch.float32)).sample  # [1, 3, 1, H, W] in ~[-1, 1]
    decoded = decoded[0, :, 0]  # [3, H, W]
    _dump(decoded, args.out / "decoded.bin")
    (args.out / "meta.txt").write_text(f"{Z_DIM} {h_in} {w_in}\n")
    print(
        f"[py-qwen-vae] done: z_dim={Z_DIM} latent={h_in}x{w_in} "
        f"decoded{tuple(decoded.shape)} range[{decoded.min():.3f},{decoded.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
