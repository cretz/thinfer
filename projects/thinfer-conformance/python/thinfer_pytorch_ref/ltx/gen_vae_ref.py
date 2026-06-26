"""LTX-2.3 video VAE decode reference (upstream `VideoDecoder`, decoder only).

Builds the upstream `ltx_core` `VideoDecoder` from the on-disk safetensors
`__metadata__.config` (the authoritative config: `timestep_conditioning=false`,
`causal_decoder=false`, `norm_layer=pixel_norm`, `patch_size=4`), loads the
bf16 `decoder.*` + `per_channel_statistics.*` weights (the SAME bytes the engine
loads; bf16 values are exact in f32, so f32 compute here == engine bf16-storage /
f32-compute), and decodes a fixed-seed NORMALIZED latent. The decoder
un-normalizes the latent internally (per-channel stats), exactly as the engine
does host-side, so both consume the same normalized latent.

Dumps (LE f32 unless noted):
  latent.bin    [128, f, h, w]   normalized latent (engine input)
  mean.bin      [128]            per_channel_statistics mean-of-means
  std.bin       [128]            per_channel_statistics std-of-means
  conv_in.bin   [1024, f, h, w]  conv_in output (post un-normalize)
  up_NN.bin     [C, T, H, W]     each up_block output (NN = 00..)
  conv_out.bin  [48, F, H, W]    conv_out output (pre 4x4 unpatchify)
  video.bin     [3, F, 32h, 32w] final decoder output (raw, no clamp)
  meta.txt      "f h w n_up F Hout Wout"

  uv run --with einops python -m thinfer_pytorch_ref.ltx.gen_vae_ref \\
      --vae <...video_vae.safetensors> --out <dir> [--frames 2 --height 4 --width 4]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def _locate_ltx_core() -> str:
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "third-party" / "LTX-2" / "packages" / "ltx-core" / "src"
        if cand.is_dir():
            return str(cand)
    raise AssertionError("could not locate third-party/LTX-2 above gen_vae_ref.py")


def _read_safetensors(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        data_start = 8 + n
        blob = fh.read()
    meta = json.loads(header["__metadata__"]["config"])
    dtype_map = {
        "F32": (torch.float32, np.float32),
        "F16": (torch.float16, np.float16),
        "BF16": (torch.bfloat16, None),
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = blob[a:b]
        tdt, ndt = dtype_map[info["dtype"]]
        if info["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)
            t = torch.from_numpy(f32.copy()).reshape(info["shape"])
        else:
            t = torch.from_numpy(np.frombuffer(raw, dtype=ndt).copy()).reshape(info["shape"])
            t = t.to(torch.float32)
        tensors[name] = t
    return meta, tensors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=2, help="latent frames f")
    p.add_argument("--height", type=int, default=4, help="latent height h")
    p.add_argument("--width", type=int, default=4, help="latent width w")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_ltx_core())
    from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator

    meta, tensors = _read_safetensors(args.vae)

    decoder = VideoDecoderConfigurator.from_config(meta).eval().float()

    # State dict: strip the `decoder.` prefix; keep `per_channel_statistics.*`.
    sd: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("decoder."):
            sd[k[len("decoder.") :]] = v
        elif k.startswith("per_channel_statistics."):
            sd[k] = v
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    # PixelNorm / norm3 / shortcut are weightless or Identity here -> no params.
    assert not [m for m in missing if "num_batches" not in m], f"missing params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    f, h, w = args.frames, args.height, args.width
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 128, f, h, w, generator=g, dtype=torch.float32)

    # Per-stage taps via forward hooks.
    taps: dict[str, torch.Tensor] = {}
    handles = []
    handles.append(decoder.conv_in.register_forward_hook(lambda m, i, o: taps.__setitem__("conv_in", o.detach())))
    handles.append(decoder.conv_out.register_forward_hook(lambda m, i, o: taps.__setitem__("conv_out", o.detach())))
    up_outs: list[torch.Tensor] = []
    for blk in decoder.up_blocks:
        blk.register_forward_hook(lambda m, i, o: up_outs.append(o.detach()))

    with torch.no_grad():
        video = decoder(latent)  # [1, 3, F, 32h, 32w], raw (no clamp)

    for hd in handles:
        hd.remove()

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("latent.bin", latent[0])  # [128, f, h, w]
    dump("mean.bin", decoder.per_channel_statistics.get_buffer("mean-of-means"))
    dump("std.bin", decoder.per_channel_statistics.get_buffer("std-of-means"))
    dump("conv_in.bin", taps["conv_in"][0])
    dump("conv_out.bin", taps["conv_out"][0])
    for i, o in enumerate(up_outs):
        dump(f"up_{i:02d}.bin", o[0])
    dump("video.bin", video[0])

    fout, hout, wout = video.shape[2], video.shape[3], video.shape[4]
    (args.out / "meta.txt").write_text(f"{f} {h} {w} {len(up_outs)} {fout} {hout} {wout}\n")
    print(f"ltx vae ref: latent[128,{f},{h},{w}] -> video[3,{fout},{hout},{wout}], {len(up_outs)} up_blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
