"""MiniMax-H3 audio VAE ENCODER reference (Ref2VA audio references): ComfyUI's
`comfy/ldm/minimax/audio_vae.py` `MiniMaxH3AudioVAE.encode` (DAC tower ->
`pre_block` attention projection -> `mean_proj` -> normalize; the posterior
MEAN, as FastVideo's `latent_dist.mode()` + `normalize_latents`), loaded
strict over the SAME Comfy-Org fp32 file the engine reads, run fp32 on CPU
(the file's own dtype, 0.6 GB). Only `comfy.ops` is shimmed to plain torch.

Input: a deterministic 2.01 s stereo test signal at 32 kHz (harmonic tones
with vibrato plus seeded noise; an odd length, so the hop padding is
exercised).

Dumps (little-endian f32): `input.bin` (`[2, L]`), `tower.bin` (channel 0's
encoder tower output, `[2048, T]`), `latents.bin` (normalized latents as DiT
rows, `[2, T, 32]` channel-major), `meta.txt` (`L T`).
"""

from __future__ import annotations

import argparse
import math
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gen_vae_ref as base

RATE = 32_000
SAMPLES = 64_333


def _shim_comfy_ops() -> None:
    def cast_to_input(w, x):
        return w.to(dtype=x.dtype, device=x.device)

    ops = types.SimpleNamespace(
        disable_weight_init=types.SimpleNamespace(
            Conv1d=nn.Conv1d,
            ConvTranspose1d=nn.ConvTranspose1d,
            Linear=nn.Linear,
            LayerNorm=nn.LayerNorm,
        ),
        cast_to_input=cast_to_input,
        cast_bias_weight=lambda layer, x, offloadable=False: (layer.weight, None, None),
        uncast_bias_weight=lambda *a, **k: None,
        scaled_dot_product_attention=F.scaled_dot_product_attention,
    )
    comfy = base._stub("comfy", __path__=[])
    comfy.ops = base._stub("comfy.ops", **vars(ops))


def signal() -> torch.Tensor:
    t = torch.arange(SAMPLES, dtype=torch.float64) / RATE
    g = torch.Generator().manual_seed(7)
    chans = []
    for f0, vib in ((180.0, 5.0), (240.0, 3.0)):
        phase = 2 * math.pi * (f0 * t + 6.0 * torch.sin(2 * math.pi * vib * t) / (2 * math.pi * vib))
        x = sum(0.3 / k * torch.sin(k * phase) for k in range(1, 6))
        x = x * (0.6 + 0.4 * torch.sin(2 * math.pi * 1.5 * t))
        x = x + 0.02 * torch.randn(SAMPLES, generator=g, dtype=torch.float64)
        chans.append(x)
    return torch.stack(chans).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    from safetensors.torch import load_file

    _shim_comfy_ops()
    mod = base._load(
        "comfy_minimax_audio_vae",
        base._third_party("comfyui") / "comfy/ldm/minimax/audio_vae.py",
    )
    model = mod.MiniMaxH3AudioVAE()
    model.load_state_dict(load_file(str(args.vae)), strict=True)
    model.eval()

    wave = signal()
    hop = model.hop_length
    x = F.pad(wave, (0, math.ceil(SAMPLES / hop) * hop - SAMPLES)).reshape(2, 1, -1)
    tower = model.encoder(x)  # [2, 2048, T]
    z = model.mean_proj(model.pre_block(tower.transpose(1, 2)).transpose(1, 2))
    z = (z - model.latents_mean.view(1, -1, 1)) / model.latents_std.view(1, -1, 1)
    # Cross-check against the module's own `encode` ([B, 32, 2, T]).
    ref = model.encode(wave[None])[0].permute(1, 0, 2)
    assert torch.allclose(ref, z, rtol=1e-5, atol=1e-5), "manual chain diverged from encode()"

    t_lat = z.shape[-1]
    wave.numpy().astype("<f4").tofile(args.out / "input.bin")
    tower[0].numpy().astype("<f4").tofile(args.out / "tower.bin")
    z.transpose(1, 2).contiguous().numpy().astype("<f4").tofile(args.out / "latents.bin")
    (args.out / "meta.txt").write_text(f"{SAMPLES} {t_lat}\n")
    print(
        f"audio enc ref: L={SAMPLES} T={t_lat} tower rms={tower.pow(2).mean().sqrt():.4f} "
        f"latents rms={z.pow(2).mean().sqrt():.4f} std={np.std(z.numpy()):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
