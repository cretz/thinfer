"""MiniMax-H3 video VAE reference: FastVideo's torch `AutoencoderKLMiniMaxH3`
(encoder + ViT decoder) over the SAME Comfy-Org fp16 VAE file the engine reads,
run in bf16 on CPU.

A synthetic moving scene is ENCODED (posterior mean) to give a natural latent,
then decoded. The engine's decoder must match the torch decode, and both must
resemble the input, which a pure parity diff could not show.

FastVideo's framework modules (envs, attention backends, platform, profiler,
config bases) are stubbed so only the two model/config files are imported from
the clone at `third-party/FastVideo`.

Dumps (little-endian f32): `latents.bin` ([24, T, h, w], DiT space, i.e.
normalized by the checkpoint's latents_mean/std), `input.bin` and `decoded.bin`
([3, F, H, W] pixels in [0, 1]), `head{d}.bin` (the first clip's output head
after d ViT blocks), `meta.txt` (`T h w F H W`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

HEIGHT, WIDTH, FRAMES = 256, 384, 34
HEAD_DEPTHS = (0, 1, 2, 6, 18, 36)


def _third_party(name: str) -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "third-party" / name
        if cand.is_dir():
            return cand
    raise SystemExit(f"third-party/{name} clone not found above this file")


def _import_converter():
    """The official released-checkpoint -> diffusers key converter (minimax-h3-mlx
    `reference/diffusers`). Its top-level diffusers imports are only used by the
    audio path, so they are stubbed."""
    _stub("diffusers", __path__=[])
    _stub("diffusers.models", __path__=[])
    _stub("diffusers.models.autoencoders", __path__=[])
    _stub("diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio", AutoencoderKLMiniMaxH3Audio=None)
    _stub("diffusers.utils", __path__=[])
    _stub("diffusers.utils.constants", SAFE_WEIGHTS_INDEX_NAME="")
    path = _third_party("minimax-h3-mlx") / "reference/diffusers/convert_minimax_h3_to_diffusers.py"
    return _load("convert_minimax_h3_to_diffusers", path)


def _stub(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_vae():
    root = _third_party("FastVideo")

    class _Env:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    for pkg in ["fastvideo", "fastvideo.configs", "fastvideo.configs.models",
                "fastvideo.configs.models.vaes", "fastvideo.models", "fastvideo.models.vaes"]:
        _stub(pkg, __path__=[])
    _stub("fastvideo.envs", FASTVIDEO_H3_VAE_INT8_SHARED_QKV=_Env(False), FASTVIDEO_H3_VAE_TILE_BATCH=_Env(1))
    _stub("fastvideo.attention", get_attn_backend=lambda *a, **k: None)
    _stub(
        "fastvideo.platforms",
        AttentionBackendEnum=enum.Enum("AttentionBackendEnum", "TORCH_SDPA FLASH_ATTN"),
        current_platform=types.SimpleNamespace(is_cuda_alike=lambda: False),
    )
    _stub("fastvideo.profiler", nvtx_range=lambda *a, **k: contextlib.nullcontext())

    @dataclasses.dataclass
    class VAEArchConfig:
        pass

    # FastVideo's production defaults: spatial tiling ON (256px, the H3 config's
    # 64px overlaps). Attention stays within a tile, so this is a different
    # computation from an untiled decode, and the engine mirrors production.
    @dataclasses.dataclass
    class VAEConfig:
        use_tiling: bool = True
        tile_sample_min_height: int = 256
        tile_sample_min_width: int = 256

    _stub("fastvideo.configs.models.vaes.base", VAEArchConfig=VAEArchConfig, VAEConfig=VAEConfig)
    cfg = _load("fastvideo.configs.models.vaes.minimax_h3_video",
                root / "fastvideo/configs/models/vaes/minimax_h3_video.py")
    vae = _load("fastvideo.models.vaes.minimax_h3_video", root / "fastvideo/models/vaes/minimax_h3_video.py")
    return cfg, vae


def synthetic_video() -> torch.Tensor:
    """[1, 3, F, H, W] in [0, 1]: sky gradient, ground, a disk crossing the frame."""
    y = torch.linspace(0, 1, HEIGHT)[:, None].expand(HEIGHT, WIDTH)
    x = torch.linspace(0, 1, WIDTH)[None, :].expand(HEIGHT, WIDTH)
    frames = []
    for f in range(FRAMES):
        r = 0.35 + 0.5 * (1 - y)
        g = 0.55 + 0.35 * (1 - y)
        b = 0.95 - 0.2 * y
        ground = y > 0.7
        r = torch.where(ground, 0.25 + 0.1 * x, r)
        g = torch.where(ground, 0.5 + 0.1 * torch.sin(20 * x), g)
        b = torch.where(ground, 0.15 + 0.0 * x, b)
        cx, cy = 0.15 + 0.7 * f / (FRAMES - 1), 0.45
        disk = ((x - cx) * WIDTH / HEIGHT) ** 2 + (y - cy) ** 2 < 0.12**2
        r = torch.where(disk, torch.full_like(r, 0.9), r)
        g = torch.where(disk, torch.full_like(g, 0.2), g)
        b = torch.where(disk, torch.full_like(b, 0.1), b)
        frames.append(torch.stack([r, g, b]))
    return torch.stack(frames, dim=1)[None].clamp(0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    from safetensors.torch import load_file

    cfg_mod, vae_mod = _import_vae()
    conv = _import_converter()
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
    assert all(k in ("latents_mean", "latents_std", "pixel_mean", "pixel_std") for k in missing), missing
    del sd
    # The class pins its modules to f32 (`_keep_in_fp32_modules`); cast every
    # parameter directly so the whole reference runs bf16 (RAM: no f32 pyref).
    for prm in model.parameters():
        prm.data = prm.data.bfloat16()
    model.eval()
    for name in ("latents_mean", "latents_std", "pixel_mean", "pixel_std"):
        setattr(model, name, getattr(model, name).float())

    video = synthetic_video()
    moments = model._encode(model.normalize_pixels(video).bfloat16()).float()
    z = moments[:, : moments.shape[1] // 2]
    latents = (z - model.latents_mean) / model.latents_std
    decoded = model.denormalize_pixels(model.decode(z.bfloat16()).sample.float()).clamp(0, 1)
    print("latents", tuple(latents.shape), "decoded", tuple(decoded.shape), flush=True)
    print("recon mse vs input", float((decoded - video[:, :, : decoded.shape[2]]).pow(2).mean()), flush=True)

    def dump(name, t):
        t[0].numpy().astype("<f4").tofile(args.out / f"{name}.bin")

    # Depth ladder over the first clip's ViT pass: `head{d}` is the output head
    # applied after d blocks, `[tokens, proj_out_dim]` (suffix dropped), so a
    # truncated engine pass is directly comparable.
    dec = model.decoder
    clip = model.post_quant_conv(z[:, :, : model.tokens_chunk_size + model.token_overlap].bfloat16())
    states = []
    hooks = [dec.transformer_blocks[0].register_forward_pre_hook(lambda m, i: states.append(i[0]))]
    hooks += [blk.register_forward_hook(lambda m, i, o: states.append(o)) for blk in dec.transformer_blocks]
    # Block-0 sublayer outputs (all tokens incl. suffix): attention after to_out,
    # and the feed-forward, each before its learned residual scale.
    sub = {}
    blk0 = dec.transformer_blocks[0]
    hooks.append(blk0.attn.register_forward_hook(lambda m, i, o: sub.setdefault("b0_attn", o)))
    hooks.append(blk0.ff.register_forward_hook(lambda m, i, o: sub.setdefault("b0_ff", o)))
    dec(clip)
    for hk in hooks:
        hk.remove()
    n_patches = clip.shape[2] * clip.shape[3] * clip.shape[4]
    for name, t in sub.items():
        dump(name, t.float())
    for d in HEAD_DEPTHS:
        dump(f"head{d}", dec.proj_out(dec.norm_out(states[d]))[:, :n_patches].float())

    dump("latents", latents)
    dump("input", video[:, :, : decoded.shape[2]])
    dump("decoded", decoded)
    _, _, t, h, w = latents.shape
    _, _, f, hh, ww = decoded.shape
    (args.out / "meta.txt").write_text(f"{t} {h} {w} {f} {hh} {ww}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
