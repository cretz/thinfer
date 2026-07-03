"""AnyFlow dual-timestep condition-embedder reference.

The AnyFlow port's ONLY new math vs the parity-gated Wan machinery is the
`AnyFlowDualTimestepTextImageEmbedding` time path: two TimestepEmbedding MLPs
(t and r) blended `(1-g)*temb + g*delta_emb` (g = 0.25, `deltatime_type="r"`),
then `time_proj(silu(rt_emb))`. This script computes that path in fp32 from
the checkpoint's own `condition_embedder.*` tensors (a few hundred MB read
from shard 1; the 28GB model is never materialized -- the low-RAM pyref
policy) and dumps `temb.bin` / `timestep_proj.bin` (f32 LE) for the engine
parity test.

Faithful to diffusers-main `transformer_anyflow.py::forward_timestep` with
batch=1, num_frames=1 (the plain-t2v scalar broadcast).

Usage:
  uv run python -m thinfer_pytorch_ref.wan.gen_anyflow_embedder_ref \
      --shard <diffusion_pytorch_model-00001-of-00003.safetensors> \
      --t 1000 --r 833.3333 --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

GATE = 0.25
FREQ_DIM = 256


def sincos(t: float) -> torch.Tensor:
    # diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0 ->
    # emb = cat(cos, sin), omega[i] = 1/10000^(i/half), computed in fp32 from
    # a float64 argument (matches the engine's f64 trig).
    half = FREQ_DIM // 2
    i = torch.arange(half, dtype=torch.float64)
    w = 1.0 / torch.pow(torch.tensor(10000.0, dtype=torch.float64), i / half)
    arg = float(t) * w
    return torch.cat([torch.cos(arg), torch.sin(arg)]).to(torch.float32)


def mlp(x: torch.Tensor, w1, b1, w2, b2) -> torch.Tensor:
    h = torch.nn.functional.silu(x @ w1.T + b1)
    return h @ w2.T + b2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, action="append",
                    help="safetensors shard; repeat for a sharded set")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors import safe_open

    pre = "condition_embedder."
    names = {
        "tw1": "time_embedder.linear_1.weight",
        "tb1": "time_embedder.linear_1.bias",
        "tw2": "time_embedder.linear_2.weight",
        "tb2": "time_embedder.linear_2.bias",
        "dw1": "delta_embedder.linear_1.weight",
        "db1": "delta_embedder.linear_1.bias",
        "dw2": "delta_embedder.linear_2.weight",
        "db2": "delta_embedder.linear_2.bias",
        "pw": "time_proj.weight",
        "pb": "time_proj.bias",
    }
    w: dict[str, torch.Tensor] = {}
    for shard in args.shard:
        with safe_open(shard, framework="pt") as f:
            have = set(f.keys())
            for k, n in names.items():
                if k not in w and pre + n in have:
                    w[k] = f.get_tensor(pre + n).to(torch.float32)
    missing = [n for k, n in names.items() if k not in w]
    assert not missing, f"embedder tensors not found in shards: {missing}"

    temb_t = mlp(sincos(args.t), w["tw1"], w["tb1"], w["tw2"], w["tb2"])
    delta = mlp(sincos(args.r), w["dw1"], w["db1"], w["dw2"], w["db2"])
    rt_emb = (1.0 - GATE) * temb_t + GATE * delta
    tproj = torch.nn.functional.silu(rt_emb) @ w["pw"].T + w["pb"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "temb.bin").write_bytes(rt_emb.numpy().astype(np.float32).tobytes())
    (out / "timestep_proj.bin").write_bytes(tproj.numpy().astype(np.float32).tobytes())
    print(f"temb std={rt_emb.std():.6f} tproj std={tproj.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
