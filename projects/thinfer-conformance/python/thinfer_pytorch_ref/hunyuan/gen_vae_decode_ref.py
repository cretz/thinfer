"""HunyuanVideo 1.5 VAE decode reference (`AutoencoderKLConv3D` decoder only).

Builds the upstream `hyvideo` `Decoder` (causal-conv3d, Wan-VAE residual family;
16x spatial / 4x temporal, z32, block_out [128,256,512,1024,1024]) directly from
the cloned reference, loads the F16 `decoder.*` weights (the SAME bytes the engine
loads; F16 values are exact in f32), and decodes a fixed-seed latent. The decoder
network input is `latent / scaling_factor` (the pipeline pre-scale); the engine
applies the same host-side scalar before its decode, so both consume the same
latent.bin and run the identical conv network.

The Hunyuan decoder DIVERGES from the engine's Wan decoder in two ways the engine
fork must reproduce (see hunyuan-plan.md): mid.attn_1 is CAUSAL spatio-temporal
(frame i attends 0..i over f*h*w tokens), and Upsample = one causal conv ->
out*factor then pixelshuffle (first latent frame spatial-only). Those taps are
dumped per-stage for bisection.

Dumps (LE f32):
  latent.bin    [32, f, h, w]    pre-scale latent (engine input)
  conv_in.bin   [1024, f, h, w]  conv_in + repeat_interleave residual
  mid.bin       [1024, f, h, w]  post mid block
  up_NN.bin     [C, T, H, W]     each up-stage output (after its upsampler)
  conv_out.bin  [3, F, Hout, Wout] conv_out (== video, raw [-1,1], no clamp)
  video.bin     [3, F, Hout, Wout] final decoder output (raw)
  meta.txt      "f h w n_up F Hout Wout"

  uv run --with einops --with loguru python -m \\
      thinfer_pytorch_ref.hunyuan.gen_vae_decode_ref \\
      --vae <...hunyuanvideo15_vae_fp16.safetensors> --out <dir> \\
      [--frames 1 --height 8 --width 8]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# Fixed config for this checkpoint (block_out reversed for the decoder).
Z_CHANNELS = 32
OUT_CHANNELS = 3
BLOCK_OUT_REVERSED = [1024, 1024, 512, 256, 128]
NUM_RES_BLOCKS = 2
FFACTOR_SPATIAL = 16
FFACTOR_TEMPORAL = 4
SCALING_FACTOR = 1.03682


def _locate_hyvideo() -> str:
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "third-party" / "HunyuanVideo-1.5"
        if (cand / "hyvideo").is_dir():
            return str(cand)
    raise AssertionError("could not locate third-party/HunyuanVideo-1.5 above this script")


def _read_safetensors(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        blob = fh.read()
    dtype_map = {"F32": np.float32, "F16": np.float16, "BF16": None}
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__" or not name.startswith(prefix):
            continue
        a, b = info["data_offsets"]
        raw = blob[a:b]
        if info["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)
            t = torch.from_numpy(f32.copy()).reshape(info["shape"])
        else:
            ndt = dtype_map[info["dtype"]]
            t = torch.from_numpy(np.frombuffer(raw, dtype=ndt).copy()).reshape(info["shape"])
            t = t.to(torch.float32)
        tensors[name[len(prefix) :]] = t
    return tensors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frames", type=int, default=1, help="latent frames f")
    p.add_argument("--height", type=int, default=8, help="latent height h")
    p.add_argument("--width", type=int, default=8, help="latent width w")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_hyvideo())
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae import Decoder

    decoder = Decoder(
        z_channels=Z_CHANNELS,
        out_channels=OUT_CHANNELS,
        block_out_channels=BLOCK_OUT_REVERSED,
        num_res_blocks=NUM_RES_BLOCKS,
        ffactor_spatial=FFACTOR_SPATIAL,
        ffactor_temporal=FFACTOR_TEMPORAL,
    ).eval().float()

    sd = _read_safetensors(args.vae, "decoder.")
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    assert not missing, f"missing params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    f, h, w = args.frames, args.height, args.width
    g = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, Z_CHANNELS, f, h, w, generator=g, dtype=torch.float32)

    # Per-stage taps via forward hooks.
    taps: dict[str, torch.Tensor] = {}
    handles = [
        decoder.conv_in.register_forward_hook(lambda m, i, o: taps.__setitem__("conv_in_raw", o.detach())),
        decoder.mid.block_1.register_forward_hook(lambda m, i, o: taps.__setitem__("mid_block1", o.detach())),
        decoder.mid.attn_1.register_forward_hook(lambda m, i, o: taps.__setitem__("mid_attn", o.detach())),
        decoder.mid.block_2.register_forward_hook(lambda m, i, o: taps.__setitem__("mid", o.detach())),
        decoder.conv_out.register_forward_hook(lambda m, i, o: taps.__setitem__("conv_out", o.detach())),
    ]
    # Each up-stage is a bare nn.Module whose forward is never called (the
    # decoder invokes its .block[j]/.upsample submodules directly), so tap the
    # stage's last executed leaf: its upsampler when present, else its last
    # resnet (the no-upsampler final stage).
    up_outs: list[torch.Tensor] = []
    for stage in decoder.up:
        leaf = stage.upsample if hasattr(stage, "upsample") else stage.block[-1]
        leaf.register_forward_hook(lambda m, i, o, _o=up_outs: _o.append(o.detach()))

    with torch.no_grad():
        video = decoder(latent / SCALING_FACTOR)  # [1, 3, F, Hout, Wout], raw

    for hd in handles:
        hd.remove()

    def dump(name: str, t: torch.Tensor) -> None:
        arr = t.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    # conv_in tap: re-add the repeat_interleave residual the hook output omits
    # (the hook captures conv_in's conv output only; the decoder adds the
    # repeat_interleave shortcut right after). Recompute it so the engine's
    # post-residual conv_in is what we compare.
    repeats = BLOCK_OUT_REVERSED[0] // Z_CHANNELS
    z_in = latent / SCALING_FACTOR
    conv_in_full = taps["conv_in_raw"] + z_in.repeat_interleave(repeats, dim=1)

    dump("latent.bin", latent[0])
    dump("conv_in.bin", conv_in_full[0])
    dump("mid_block1.bin", taps["mid_block1"][0])
    dump("mid_attn.bin", taps["mid_attn"][0])
    dump("mid.bin", taps["mid"][0])
    for i, o in enumerate(up_outs):
        dump(f"up_{i:02d}.bin", o[0])
    dump("conv_out.bin", taps["conv_out"][0])
    dump("video.bin", video[0])

    fout, hout, wout = video.shape[2], video.shape[3], video.shape[4]
    (args.out / "meta.txt").write_text(f"{f} {h} {w} {len(up_outs)} {fout} {hout} {wout}\n")
    print(f"hunyuan vae ref: latent[32,{f},{h},{w}] -> video[3,{fout},{hout},{wout}], {len(up_outs)} up stages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
