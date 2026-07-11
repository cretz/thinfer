"""LTX-2.3 audio VAE DECODER reference (upstream `AudioDecoder`, latent -> mel).

Builds the upstream `ltx_core` `AudioDecoder` from the on-disk audio VAE
safetensors `__metadata__.config` (`ch 128, ch_mult [1,2,4], z 8, mel_bins 64,
norm pixel, causality_axis height, no attention`), loads the bf16
`audio_vae.decoder.*` + `audio_vae.per_channel_statistics.*` weights (the SAME
bytes the engine loads; bf16 is exact in f32), and decodes a fixed-seed latent
`[1,8,frames,16]` to a mel `[1,2,4*frames-3,64]`. CPU-only (the decoder is tiny);
no vocoder here (that is a separate, larger ref).

Dumps (LE f32):
  latent.bin   [8, frames, 16]      input latent
  mel.bin      [2, 4*frames-3, 64]  decoded mel (vocoder input)
  meta.txt     "frames Tmel mel_bins"

  uv run --with einops python -m thinfer_pytorch_ref.ltx.gen_audio_vae_ref \\
      --audio-vae <...audio_vae.safetensors> --out <dir> [--frames 2]
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
    raise AssertionError("could not locate third-party/LTX-2 above gen_audio_vae_ref.py")


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
        else:
            raise AssertionError(f"unexpected dtype {info['dtype']}")
        tensors[name] = t.to(torch.float32)
    return meta, tensors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=2, help="latent frames")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_ltx_core())
    from ltx_core.model.audio_vae import AudioDecoderConfigurator

    meta, tensors = _read_safetensors(args.audio_vae)
    decoder = AudioDecoderConfigurator.from_config(meta).eval().float()

    # State dict: strip `audio_vae.decoder.`; keep `per_channel_statistics.*`.
    sd: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("audio_vae.decoder."):
            sd[k[len("audio_vae.decoder.") :]] = v
        elif k.startswith("audio_vae.per_channel_statistics."):
            sd["per_channel_statistics." + k[len("audio_vae.per_channel_statistics.") :]] = v
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "num_batches" not in m], f"missing params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    f = args.frames
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 8, f, 16, generator=g, dtype=torch.float32)

    with torch.no_grad():
        mel = decoder(latent)  # [1, 2, 4*f-3, 64]

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("latent.bin", latent[0])
    dump("mel.bin", mel[0])
    tmel, mel_bins = mel.shape[2], mel.shape[3]
    (args.out / "meta.txt").write_text(f"{f} {tmel} {mel_bins}\n")
    print(f"ltx audio-vae ref: latent[8,{f},16] -> mel[2,{tmel},{mel_bins}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
