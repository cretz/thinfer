"""MiniMax-H3 text conditioner reference: Qwen3-VL-32B layers 0..49 over the SAME
Q4_K_M GGUF the engine reads (dequantized to f32 one layer at a time, so peak RAM
stays at one layer).

H3 conditions on `hidden_states[50]`: the residual stream after decoder layer 49,
WITHOUT the final norm. Text-only M-RoPE with equal t/h/w positions is plain 1-D
rope (half-rot), causal attention, no chat template, no special tokens.

Dumps (little-endian): `prompt.txt`, `ids.bin` (i32), `embed.bin`, `layer{i}.bin`
(the state after decoder layer i) for a few probe layers, `hidden.bin` ([seq,
5120] f32), `meta.txt` (`seq`).

Run:
  uv run --with gguf --with tokenizers python -m thinfer_pytorch_ref.minimax.gen_encoder_ref \\
      --gguf <qwen3vl...Q4_K_M.gguf> --tokenizer <tokenizer.json> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

N_LAYERS = 50
HIDDEN = 5120
N_HEADS = 64
N_KV = 8
HEAD_DIM = 128
EPS = 1e-6
ROPE_THETA = 5_000_000.0
PROBE_LAYERS = (0, 1, 2, 5, 10, 20, 30, 40)
PROMPT = (
    "A red fox trots across a snowy meadow at dawn, its breath steaming in the cold air, "
    "while birds call from the pine trees."
)


class Weights:
    def __init__(self, path: Path):
        import gguf

        self._q = gguf.quants
        self.tensors = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}

    def get(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        arr = np.ascontiguousarray(self._q.dequantize(t.data, t.tensor_type), dtype=np.float32)
        shape = [int(s) for s in reversed(t.shape)]
        return torch.from_numpy(arr.reshape(shape))

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        t = self.tensors[name]
        arr = np.ascontiguousarray(self._q.dequantize(t.data[ids], t.tensor_type), dtype=np.float32)
        return torch.from_numpy(arr.reshape(len(ids), -1))


def rms(x, w):
    return F.rms_norm(x, (x.shape[-1],), w, EPS)


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def layer(W: Weights, i: int, x, cos, sin, mask):
    p = f"model.layers.{i}."
    seq = x.shape[0]
    h = rms(x, W.get(p + "input_layernorm.weight"))
    q = (h @ W.get(p + "self_attn.q_proj.weight").T).view(seq, N_HEADS, HEAD_DIM)
    k = (h @ W.get(p + "self_attn.k_proj.weight").T).view(seq, N_KV, HEAD_DIM)
    v = (h @ W.get(p + "self_attn.v_proj.weight").T).view(seq, N_KV, HEAD_DIM)
    q = rope(rms(q, W.get(p + "self_attn.q_norm.weight")), cos, sin)
    k = rope(rms(k, W.get(p + "self_attn.k_norm.weight")), cos, sin)
    rep = N_HEADS // N_KV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    a = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
    )
    a = a.transpose(0, 1).reshape(seq, N_HEADS * HEAD_DIM)
    x = x + a @ W.get(p + "self_attn.o_proj.weight").T
    h = rms(x, W.get(p + "post_attention_layernorm.weight"))
    g = h @ W.get(p + "mlp.gate_proj.weight").T
    u = h @ W.get(p + "mlp.up_proj.weight").T
    return x + (F.silu(g) * u) @ W.get(p + "mlp.down_proj.weight").T


def dump(out: Path, name: str, t) -> None:
    arr = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    arr.astype("<f4" if arr.dtype.kind == "f" else "<i4").tofile(out / f"{name}.bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    from tokenizers import Tokenizer

    ids = Tokenizer.from_file(str(args.tokenizer)).encode(PROMPT, add_special_tokens=False).ids
    seq = len(ids)
    dump(args.out, "ids", np.asarray(ids, dtype=np.int32))
    (args.out / "prompt.txt").write_text(PROMPT, encoding="utf-8")

    W = Weights(args.gguf)
    x = W.rows("model.embed_tokens.weight", ids)
    dump(args.out, "embed", x)

    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    f = torch.arange(seq, dtype=torch.float64)[:, None] * inv[None, :]
    f = torch.cat((f, f), dim=-1)
    cos = f.cos().float()[:, None, :]
    sin = f.sin().float()[:, None, :]
    mask = torch.ones(seq, seq, dtype=torch.bool).tril()

    for i in range(N_LAYERS):
        x = layer(W, i, x, cos, sin, mask)
        print(f"layer {i:2d}: absmax {x.abs().max():10.2f} rms {x.pow(2).mean().sqrt():8.3f}", flush=True)
        if i in PROBE_LAYERS:
            dump(args.out, f"layer{i}", x)
    dump(args.out, "hidden", x)
    (args.out / "meta.txt").write_text(f"{seq}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
