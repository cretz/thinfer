"""MiniMax-H3 video VAE ENCODER reference, single frame (a keyframe or an image
reference): FastVideo's torch `AutoencoderKLMiniMaxH3._encode_clip` (the
single-frame path, no temporal chunking) with tiling OFF, over the SAME
Comfy-Org fp16 VAE file the engine reads, run in bf16 on CPU. Reuses
`gen_vae_ref`'s model construction.

Dumps (little-endian f32, batch dropped): `input.bin` ([3, H, W] pixels in
[0, 1]), `conv_in.bin` and `down{i}.bin` (the stage outputs, `[C, H', W']`),
`enc_out.bin` (the encoder output, `[48, h, w]`), `moments.bin` (after
`quant_conv`, `[48, h, w]`: mean then logvar), `latents.bin` (the posterior
MEAN normalized by latents_mean / std, `[24, h, w]`), `meta.txt` (`H W h w`).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from . import gen_vae_ref as base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    from safetensors.torch import load_file

    cfg_mod, vae_mod = base._import_vae()
    conv = base._import_converter()
    raw = load_file(str(args.vae))
    mean = raw["latents_mean"].float()
    std = raw["latents_std"].float()
    config = cfg_mod.MiniMaxH3VideoVAEConfig()
    arch = dataclasses.asdict(config.arch_config)
    sd = {}
    for key, tensor in raw.items():
        if key in ("latents_mean", "latents_std"):
            continue
        for k, t in conv.convert_video_vae_key(key, tensor, arch):
            sd[k] = t
    del raw
    config.arch_config.latents_mean = tuple(mean.tolist())
    config.arch_config.latents_std = tuple(std.tolist())
    model = vae_mod.AutoencoderKLMiniMaxH3(config)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    del sd
    for prm in model.parameters():
        prm.data = prm.data.bfloat16()
    model.eval()
    for name in ("latents_mean", "latents_std", "pixel_mean", "pixel_std"):
        setattr(model, name, getattr(model, name).float())
    model.use_tiling = False

    frame = base.synthetic_video()[:, :, :1]  # [1, 3, 1, H, W]
    enc = model.encoder
    taps = {}
    hooks = [enc.conv_in.register_forward_hook(lambda m, i, o: taps.setdefault("conv_in", o))]
    for idx, blk in enumerate(enc.down_blocks):
        hooks.append(blk.register_forward_hook(lambda m, i, o, idx=idx: taps.setdefault(f"down{idx}", o)))
    hooks.append(enc.register_forward_hook(lambda m, i, o: taps.setdefault("enc_out", o)))
    moments = model._encode_clip(model.normalize_pixels(frame).bfloat16()).float()
    for hk in hooks:
        hk.remove()
    z = moments[:, : moments.shape[1] // 2]
    latents = (z - model.latents_mean) / model.latents_std
    print("moments", tuple(moments.shape), "latents rms", float(latents.pow(2).mean().sqrt()), flush=True)

    def dump(name, t):
        # [1, C, 1, H, W] -> [C, H, W]
        t.float()[0, :, 0].numpy().astype("<f4").tofile(args.out / f"{name}.bin")

    for name, t in taps.items():
        dump(name, t)
    dump("moments", moments)
    dump("latents", latents)
    dump("input", frame)
    _, _, _, hh, ww = frame.shape
    _, _, _, h, w = latents.shape
    (args.out / "meta.txt").write_text(f"{hh} {ww} {h} {w}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
