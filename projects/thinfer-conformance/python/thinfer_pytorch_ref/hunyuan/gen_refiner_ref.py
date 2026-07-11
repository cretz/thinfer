"""HunyuanVideo 1.5 SingleTokenRefiner reference (`txt_in` of the DiT).

Builds the upstream `SingleTokenRefiner` (in 3584, hidden 2048, 16 heads, depth 2,
mlp x4 silu, qk_norm off, qkv bias) from the cloned reference, loads the `txt_in.*`
weights out of the lightx2v 4-step DiT safetensors (the SAME bytes the engine
loads; the file is fp16, exact in f32), and refines a fixed-seed text-hidden input
at a fixed timestep with mask=None (plain mean + full bidirectional attention).

Dumps (LE f32):
  text_in.bin   [seq, 3584]   input text hidden (engine input)
  t_emb.bin     [2048]        TimestepEmbedder(t)
  c_emb.bin     [2048]        TextProjection(mean(x))
  cond.bin      [2048]        c = t_emb + c_emb
  embedded.bin  [seq, 2048]   input_embedder(x)
  block0.bin    [seq, 2048]   after refiner block 0
  refined.bin   [seq, 2048]   final output
  meta.txt      "seq"

  uv run --with einops --with loguru python -m \\
      thinfer_pytorch_ref.hunyuan.gen_refiner_ref \\
      --dit <...hy1.5_t2v_480p_lightx2v_4step.safetensors> --out <dir> [--seq 16]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

IN_CHANNELS = 3584
HIDDEN = 2048
HEADS = 16
DEPTH = 2
TIMESTEP = 500.0


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
        base = 8 + n
        out: dict[str, torch.Tensor] = {}
        dtype_map = {"F32": np.float32, "F16": np.float16}
        for name, info in header.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            a, b = info["data_offsets"]
            fh.seek(base + a)
            raw = fh.read(b - a)
            if info["dtype"] == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                f32 = (u16 << 16).view(np.float32)
                t = torch.from_numpy(f32.copy()).reshape(info["shape"])
            else:
                ndt = dtype_map[info["dtype"]]
                t = torch.from_numpy(np.frombuffer(raw, dtype=ndt).copy()).reshape(info["shape"])
                t = t.to(torch.float32)
            out[name[len(prefix):]] = t
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_hyvideo())
    from hyvideo.models.transformers.modules.token_refiner import SingleTokenRefiner

    refiner = SingleTokenRefiner(
        in_channels=IN_CHANNELS, hidden_size=HIDDEN, heads_num=HEADS, depth=DEPTH
    ).eval().float()

    sd = _read_safetensors(args.dit, "txt_in.")
    missing, unexpected = refiner.load_state_dict(sd, strict=False)
    assert not missing, f"missing params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    seq = args.seq
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(1, seq, IN_CHANNELS, generator=g, dtype=torch.float32)
    t = torch.tensor([TIMESTEP], dtype=torch.float32)

    taps: dict[str, torch.Tensor] = {}
    refiner.individual_token_refiner.blocks[0].register_forward_hook(
        lambda m, i, o: taps.__setitem__("block0", o.detach())
    )

    with torch.no_grad():
        t_emb = refiner.t_embedder(t)
        c_emb = refiner.c_embedder(x.mean(dim=1))
        cond = t_emb + c_emb
        embedded = refiner.input_embedder(x)
        refined = refiner(x, t, mask=None)

    def dump(name: str, tt: torch.Tensor) -> None:
        arr = tt.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    dump("text_in.bin", x[0])
    dump("t_emb.bin", t_emb[0])
    dump("c_emb.bin", c_emb[0])
    dump("cond.bin", cond[0])
    dump("embedded.bin", embedded[0])
    dump("block0.bin", taps["block0"][0])
    dump("refined.bin", refined[0])
    (args.out / "meta.txt").write_text(f"{seq}\n")
    print(f"hunyuan refiner ref: x[{seq},{IN_CHANNELS}] t={TIMESTEP} -> refined[{seq},{HIDDEN}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
