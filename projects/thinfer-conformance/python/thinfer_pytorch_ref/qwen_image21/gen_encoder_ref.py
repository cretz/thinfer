"""Qwen-Image-2.1 text conditioner reference: transformers' Qwen3-VL-8B text
decoder layers over the SAME GGUF the engine reads, one layer resident at a
time (f32), so peak RAM stays at one layer.

The prompt is the diffusers 2.1 template (`modular_pipelines/qwenimage21/
encoders.py`), text only. The conditioner is `hidden_states[-1]` BEFORE the
final norm; the pipeline drops the system rows (`drop` in meta.txt).

  text.txt       the templated prompt.
  ids.bin        i32 [seq]          token ids (the 2.1 processor's tokenizer).
  embed.bin      f32 [seq, 4096]    embedding gather.
  layer{i}.bin   f32 [seq, 4096]    residual stream after decoder layer i.
  hidden.bin     f32 [seq, 4096]    after the last layer, un-normed.
  meta.txt       "seq drop"

Usage:

    uv run --with gguf --with tokenizers python -m thinfer_pytorch_ref.qwen_image21.gen_encoder_ref \\
        --gguf <Qwen3-VL-8B-Instruct-Q8_0.gguf> --tokenizer <processor/tokenizer.json> \\
        --config <Qwen3-VL-8B-Instruct config.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thinfer_pytorch_ref.qwen_image21.gen_dit_ref import _dump

SYSTEM = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
PROMPT = (
    "A red fox trots across a snowy meadow at dawn, its breath steaming in the cold air, "
    "while birds call from the pine trees."
)
PROBES = (0, 1, 10)

LAYER_MAP = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


class Weights:
    def __init__(self, path: Path):
        import gguf

        self._q = gguf.quants
        self.tensors = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}

    def get(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        arr = np.ascontiguousarray(self._q.dequantize(t.data, t.tensor_type), dtype=np.float32)
        return torch.from_numpy(arr.reshape([int(s) for s in reversed(t.shape)]))

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        t = self.tensors[name]
        arr = np.ascontiguousarray(self._q.dequantize(t.data[ids], t.tensor_type), dtype=np.float32)
        return torch.from_numpy(arr.reshape(len(ids), -1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextRotaryEmbedding,
    )

    tok = Tokenizer.from_file(str(args.tokenizer))
    text = f"{SYSTEM}<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_special_tokens=False).ids
    drop = len(tok.encode(SYSTEM, add_special_tokens=False).ids)
    np.asarray(ids, dtype="<i4").tofile(str(args.out / "ids.bin"))
    (args.out / "text.txt").write_text(text, encoding="utf-8", newline="")
    seq = len(ids)

    cfg = Qwen3VLTextConfig(**json.loads(args.config.read_text())["text_config"])
    cfg._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    w = Weights(args.gguf)

    x = w.rows("token_embd.weight", ids)[None]
    _dump(x[0], args.out / "embed.bin")
    pos = torch.arange(seq)[None]
    cos_sin = Qwen3VLTextRotaryEmbedding(cfg)(x, pos)
    mask = torch.full((seq, seq), float("-inf")).triu(1)[None, None]

    layer = Qwen3VLTextDecoderLayer(cfg, 0).eval()
    params = dict(layer.named_parameters())
    for i in range(cfg.num_hidden_layers):
        for g, h in LAYER_MAP.items():
            params[f"{h}.weight"].copy_(w.get(f"blk.{i}.{g}.weight"))
        out = layer(x, position_embeddings=cos_sin, attention_mask=mask, position_ids=pos)
        x = out[0] if isinstance(out, tuple) else out
        if i in PROBES:
            _dump(x[0], args.out / f"layer{i}.bin")
    _dump(x[0], args.out / "hidden.bin")
    (args.out / "meta.txt").write_text(f"{seq} {drop}\n")
    print(f"[py-qwen21-enc] seq={seq} drop={drop} absmax={x.abs().max():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
