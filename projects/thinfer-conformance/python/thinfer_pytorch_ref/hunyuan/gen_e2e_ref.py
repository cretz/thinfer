"""HunyuanVideo 1.5 T2V end-to-end reference (denoise + VAE decode).

Pinned noise + fixed text-hidden -> 4-step flow-match Euler denoise (lightx2v
shift 9, CFG off) -> final latent -> VAE decode -> raw video. Reuses
`gen_dit_ref.dit_forward` for the per-step DiT and `gen_vae_decode_ref`'s
`Decoder` for the VAE, so the e2e reference cannot drift from the per-component
gates. The engine e2e replays the SAME `text_in.bin` + `latent_init.bin` and
compares the final latent + the decoded video.

Dumps (LE f32):
  text_in.bin     [seq, 3584]      text hidden (engine input)
  latent_init.bin [32, T, H, W]    pinned initial noise (engine input)
  latent.bin      [32, T, H, W]    final denoised latent
  video.bin       [3, F, Hpx, Wpx] VAE-decoded video (raw [-1,1])
  meta.txt        "seq T H W F Hpx Wpx"

  uv run --with einops --with loguru python -m \\
      thinfer_pytorch_ref.hunyuan.gen_e2e_ref --dit <dit.safetensors> \\
      --vae <vae.safetensors> --out <dir> [--seq 16 --t 2 --h 4 --w 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from thinfer_pytorch_ref.hunyuan.gen_dit_ref import (
    IN_CHANNELS,
    LATENT,
    _locate_hyvideo,
    _read_safetensors,
    dit_forward,
    load_dit,
)

SCALING_FACTOR = 1.03682
SHIFT = 9.0
DENOISING_STEP_LIST = [1000, 750, 500, 250]
TRAIN_STEPS = 1000
BLOCK_OUT_REVERSED = [1024, 1024, 512, 256, 128]


def schedule(shift: float, labels: list[int], train: int):
    """Mirror `FlowMatchSchedule`: returns (timesteps, sigmas[+terminal 0])."""

    def shifted(s: float) -> float:
        return shift * s / (1.0 + (shift - 1.0) * s)

    sigmas, timesteps = [], []
    for label in labels:
        idx = train - label
        s = shifted(1.0 - idx / train)
        sigmas.append(s)
        timesteps.append(s * 1000.0)
    sigmas.append(0.0)
    return timesteps, sigmas


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True, type=Path)
    p.add_argument("--vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--t", type=int, default=2)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--w", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd, refiner, mods = load_dit(args.dit)
    seq, T, H, W = args.seq, args.t, args.h, args.w
    g = torch.Generator().manual_seed(args.seed)
    text = torch.randn(1, seq, IN_CHANNELS, generator=g, dtype=torch.float32)
    latent = torch.randn(1, LATENT, T, H, W, generator=g, dtype=torch.float32)
    latent_init = latent.clone()

    timesteps, sigmas = schedule(SHIFT, DENOISING_STEP_LIST, TRAIN_STEPS)
    for i, ts in enumerate(timesteps):
        hidden = torch.cat(
            [latent, torch.zeros(1, LATENT, T, H, W), torch.zeros(1, 1, T, H, W)], dim=1
        )
        v = dit_forward(
            sd, refiner, mods, text, hidden, torch.tensor([ts], dtype=torch.float32), (T, H, W)
        )  # [1, THW, 32] token-major
        v = v.transpose(1, 2).reshape(1, LATENT, T, H, W)  # -> channel-major
        latent = latent + (sigmas[i + 1] - sigmas[i]) * v

    # VAE decode the final latent.
    sys.path.insert(0, _locate_hyvideo())
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import Decoder

    decoder = (
        Decoder(
            z_channels=LATENT,
            out_channels=3,
            block_out_channels=BLOCK_OUT_REVERSED,
            num_res_blocks=2,
            ffactor_spatial=16,
            ffactor_temporal=4,
        )
        .eval()
        .float()
    )
    vsd = _read_safetensors(args.vae, "decoder.")
    miss, unexp = decoder.load_state_dict(vsd, strict=False)
    assert not miss, f"vae missing: {miss}"
    assert not unexp, f"vae unexpected: {unexp}"
    with torch.no_grad():
        video = decoder(latent / SCALING_FACTOR)  # [1, 3, F, Hpx, Wpx]

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("text_in.bin", text[0])
    dump("latent_init.bin", latent_init[0])
    dump("latent.bin", latent[0])
    dump("video.bin", video[0])
    Fp, Hp, Wp = video.shape[2], video.shape[3], video.shape[4]
    (args.out / "meta.txt").write_text(f"{seq} {T} {H} {W} {Fp} {Hp} {Wp}\n")
    print(f"hunyuan e2e ref: seq={seq} grid=({T},{H},{W}) -> video[3,{Fp},{Hp},{Wp}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
