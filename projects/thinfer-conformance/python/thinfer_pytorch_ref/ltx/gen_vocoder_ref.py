"""LTX-2.3 vocoder reference (upstream `VocoderWithBWE`, mel -> 48kHz waveform).

Builds the upstream `ltx_core` `VocoderWithBWE` from the on-disk audio VAE
safetensors `__metadata__.config` `vocoder` block (main BigVGAN init 1536 / 6 up
stages, BWE init 512 / 5 stages, mel_stft n_fft 512 hop 80 mel 64, output 48000
Hz), loads the bf16 `vocoder.*` weights (strip leading `vocoder.` ONCE), and runs
a fixed-seed mel `[1,2,T,64]` through the full chain (main vocoder + STFT + BWE
residual + sinc resample + clamp). CPU-only; whole tail runs f32 (upstream
autocast). Used to validate the engine vocoder graph + shapes.

Dumps (LE f32):
  mel.bin    [2, T, 64]      input mel
  wav.bin    [2, T_wav]      output waveform (48kHz stereo, clamped [-1,1])
  meta.txt   "T T_wav out_sr"

  uv run --with einops --with torchaudio python -m \\
      thinfer_pytorch_ref.ltx.gen_vocoder_ref --audio-vae <...> --out <dir> [--frames 8]
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
    raise AssertionError("could not locate third-party/LTX-2 above gen_vocoder_ref.py")


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
    p.add_argument("--frames", type=int, default=8, help="mel time frames")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_ltx_core())
    from ltx_core.model.audio_vae import VocoderConfigurator

    meta, tensors = _read_safetensors(args.audio_vae)
    vocoder = VocoderConfigurator.from_config(meta).eval().float()

    # State dict: strip the leading `vocoder.` ONCE (keeps the nested bwe prefix).
    sd: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if k.startswith("vocoder."):
            sd[k[len("vocoder.") :]] = v
    missing, unexpected = vocoder.load_state_dict(sd, strict=False)
    # The Hann resampler filter is persistent=False (regenerated at construction).
    bad_missing = [m for m in missing if "resampler" not in m and "num_batches" not in m]
    assert not bad_missing, f"missing params: {bad_missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    t = args.frames
    g = torch.Generator().manual_seed(args.seed)
    mel = torch.randn(1, 2, t, 64, generator=g, dtype=torch.float32)

    with torch.no_grad():
        # main vocoder output (pre-BWE; clamped internally) for bisecting the port.
        main_out = vocoder.vocoder(mel.float())  # [1, 2, L16]
        wav = vocoder(mel)  # [1, 2, T_wav]

    def dump(name: str, x: torch.Tensor) -> None:
        arr = x.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("mel.bin", mel[0])
    dump("main.bin", main_out[0])
    dump("wav.bin", wav[0])
    t_wav = wav.shape[2]
    out_sr = meta["vocoder"]["bwe"]["output_sampling_rate"]
    (args.out / "meta.txt").write_text(f"{t} {t_wav} {out_sr}\n")
    print(f"ltx vocoder ref: mel[2,{t},64] -> wav[2,{t_wav}] @ {out_sr}Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
