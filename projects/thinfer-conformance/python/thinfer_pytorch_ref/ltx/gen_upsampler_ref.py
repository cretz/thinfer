"""LTX-2.3 latent spatial upscaler reference (upstream `LatentUpsampler`).

Builds the upstream `ltx_core` `LatentUpsampler` from the on-disk safetensors
`__metadata__.config` (`in_channels=128, mid_channels=1024, num_blocks_per_stage=
4, dims=3, spatial_upsample=true, temporal_upsample=false`), loads the bf16
weights (the SAME bytes the engine loads; bf16 is exact in f32, so f32 compute
here == engine bf16-storage / f32-compute), and upsamples a fixed-seed latent x2
spatially. The model operates on UN-normalized latents; the standalone parity
feeds a plain random latent (the normalize/un-normalize wrap is the VAE's job,
tested separately), so both sides consume the same input.

Dumps (LE f32):
  latent.bin   [128, f, h, w]      input latent (engine input)
  out.bin      [128, f, 2h, 2w]    upsampled latent (final)
  meta.txt     "f h w"

  uv run --with einops python -m thinfer_pytorch_ref.ltx.gen_upsampler_ref \\
      --upsampler <...spatial-upscaler-x2...safetensors> --out <dir> \\
      [--frames 2 --height 4 --width 4]
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
    raise AssertionError("could not locate third-party/LTX-2 above gen_upsampler_ref.py")


def _read_safetensors(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()
    meta = json.loads(header["__metadata__"]["config"])
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = blob[a:b]
        if info["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)
            t = torch.from_numpy(f32.copy()).reshape(info["shape"])
        elif info["dtype"] == "F32":
            t = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).reshape(info["shape"])
        elif info["dtype"] == "F16":
            t = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).reshape(info["shape"]).float()
        else:
            raise AssertionError(f"unexpected dtype {info['dtype']}")
        tensors[name] = t.to(torch.float32)
    return meta, tensors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upsampler", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=2, help="latent frames f")
    p.add_argument("--height", type=int, default=4, help="latent height h")
    p.add_argument("--width", type=int, default=4, help="latent width w")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_ltx_core())
    from ltx_core.model.upsampler.model import LatentUpsampler

    meta, tensors = _read_safetensors(args.upsampler)
    cfg = {k: v for k, v in meta.items() if not k.startswith("_")}
    model = LatentUpsampler(**cfg).eval().float()
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    assert not [m for m in missing if "num_batches" not in m], f"missing params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    f, h, w = args.frames, args.height, args.width
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 128, f, h, w, generator=g, dtype=torch.float32)

    with torch.no_grad():
        out = model(latent)  # [1, 128, f, 2h, 2w]

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("latent.bin", latent[0])
    dump("out.bin", out[0])
    (args.out / "meta.txt").write_text(f"{f} {h} {w}\n")
    print(
        f"ltx upsampler ref: latent[128,{f},{h},{w}] -> out[128,{out.shape[2]},{out.shape[3]},{out.shape[4]}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
