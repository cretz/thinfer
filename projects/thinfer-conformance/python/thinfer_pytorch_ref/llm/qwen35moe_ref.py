"""Qwen3.5/3.6-MoE (`qwen35moe`) text reference: the REAL transformers modules
(`Qwen3_5MoeGatedDeltaNet`, `Qwen3_5MoeAttention`, `Qwen3_5MoeSparseMoeBlock`,
`Qwen3_5MoeRMSNorm`) over the SAME GGUF the engine reads, one decoder layer
resident at a time in bf16 (peak RAM ~ one layer), 2 torch threads.

The llama.cpp converter rewrites several tensors; this file inverts them so the
HF modules see their native parameters:
  * linear-attn V heads: tiled (GGUF) -> grouped (HF) order (qkv V rows, z rows,
    a/b rows, A/dt, conv V channels, out_proj columns),
  * zero-centered norms stored +1 (all `*norm.weight` except `linear_attn.norm`),
  * `ssm_a` = -exp(A_log), conv1d squeezed, experts split gate/up.

Outputs (all f32 little-endian, row-major):
  ids.bin        i32 [T]            token ids of the prompt.
  embed.bin      f32 [T, H]         embedding gather.
  layer{i}.bin   f32 [T, H]         residual stream after decoder layer i.
  route{i}.bin   i32 [T, k]         router top-k expert ids of layer i.
  routew{i}.bin  f32 [T, k]         their routing weights.
  logits.bin     f32 [V]            last-row logits (only when all layers ran).
  meta.txt       "T H layers"

Usage:
  uv run --with gguf --with "transformers>=5.8" python -m thinfer_pytorch_ref.llm.qwen35moe_ref \
      --gguf <file.gguf> --config <config.json> --tokenizer <tokenizer.json> --out <dir> \
      [--layers N] [--prompt TEXT]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

PROMPT = "The capital of France is Paris. The capital of Germany is"


class Gguf:
    def __init__(self, path: Path):
        import gguf

        self._q = gguf.quants
        self.tensors = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}

    def get(self, name: str) -> torch.Tensor:
        """Dequantized tensor in PyTorch (outer-first) shape, f32."""
        t = self.tensors[name]
        arr = self._q.dequantize(t.data, t.tensor_type)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        return torch.from_numpy(arr.reshape([int(s) for s in reversed(t.shape)]))

    def rows(self, name: str, ids: list[int]) -> torch.Tensor:
        t = self.tensors[name]
        arr = self._q.dequantize(t.data[ids], t.tensor_type)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32).reshape(len(ids), -1))

    def has(self, name: str) -> bool:
        return name in self.tensors


def untile(x: torch.Tensor, dim: int, nk: int, nvpk: int, hd: int) -> torch.Tensor:
    """Inverse of llama.cpp `_reorder_v_heads`: tiled [nvpk, nk, hd] -> grouped [nk, nvpk, hd]."""
    dim = dim % x.dim()
    shape = list(x.shape)
    new = shape[:dim] + [nvpk, nk, hd] + shape[dim + 1 :]
    y = x.reshape(new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return y.permute(*perm).contiguous().reshape(shape)


def load_layer_state(g: Gguf, cfg, i: int) -> dict[str, torch.Tensor]:
    """GGUF blk.{i} -> HF `model.layers.{i}` state dict (decoder-layer-relative names)."""
    b = f"blk.{i}."
    sd: dict[str, torch.Tensor] = {}
    sd["input_layernorm.weight"] = g.get(b + "attn_norm.weight") - 1.0
    sd["post_attention_layernorm.weight"] = g.get(b + "post_attention_norm.weight") - 1.0
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    hk, hv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    r = nv // nk
    if cfg.layer_types[i] == "linear_attention":
        qkv = g.get(b + "attn_qkv.weight")
        qk = nk * hk * 2
        qkv = torch.cat([qkv[:qk], untile(qkv[qk:], 0, nk, r, hv)], 0)
        sd["linear_attn.in_proj_qkv.weight"] = qkv
        sd["linear_attn.in_proj_z.weight"] = untile(g.get(b + "attn_gate.weight"), 0, nk, r, hv)
        sd["linear_attn.in_proj_b.weight"] = untile(g.get(b + "ssm_beta.weight"), 0, nk, r, 1)
        sd["linear_attn.in_proj_a.weight"] = untile(g.get(b + "ssm_alpha.weight"), 0, nk, r, 1)
        a = untile(g.get(b + "ssm_a").unsqueeze(-1), 0, nk, r, 1).squeeze(-1)
        sd["linear_attn.A_log"] = torch.log(-a)
        sd["linear_attn.dt_bias"] = untile(g.get(b + "ssm_dt.bias").unsqueeze(-1), 0, nk, r, 1).squeeze(-1)
        conv = g.get(b + "ssm_conv1d.weight")  # [C, kernel]
        conv = torch.cat([conv[:qk], untile(conv[qk:], 0, nk, r, hv)], 0)
        sd["linear_attn.conv1d.weight"] = conv.unsqueeze(1)
        sd["linear_attn.norm.weight"] = g.get(b + "ssm_norm.weight")
        sd["linear_attn.out_proj.weight"] = untile(g.get(b + "ssm_out.weight"), 1, nk, r, hv)
    else:
        sd["self_attn.q_proj.weight"] = g.get(b + "attn_q.weight")
        sd["self_attn.k_proj.weight"] = g.get(b + "attn_k.weight")
        sd["self_attn.v_proj.weight"] = g.get(b + "attn_v.weight")
        sd["self_attn.o_proj.weight"] = g.get(b + "attn_output.weight")
        sd["self_attn.q_norm.weight"] = g.get(b + "attn_q_norm.weight") - 1.0
        sd["self_attn.k_norm.weight"] = g.get(b + "attn_k_norm.weight") - 1.0
    sd["mlp.gate.weight"] = g.get(b + "ffn_gate_inp.weight")
    gate = g.get(b + "ffn_gate_exps.weight")
    up = g.get(b + "ffn_up_exps.weight")
    sd["mlp.experts.gate_up_proj"] = torch.cat([gate, up], 1)
    del gate, up
    sd["mlp.experts.down_proj"] = g.get(b + "ffn_down_exps.weight")
    sd["mlp.shared_expert.gate_proj.weight"] = g.get(b + "ffn_gate_shexp.weight")
    sd["mlp.shared_expert.up_proj.weight"] = g.get(b + "ffn_up_shexp.weight")
    sd["mlp.shared_expert.down_proj.weight"] = g.get(b + "ffn_down_shexp.weight")
    sd["mlp.shared_expert_gate.weight"] = g.get(b + "ffn_gate_inp_shexp.weight").reshape(1, -1)
    return sd


def dump(path: Path, t: torch.Tensor) -> None:
    path.write_bytes(t.detach().to(torch.float32).contiguous().numpy().tobytes())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", type=int, default=0, help="0 = all")
    p.add_argument("--prompt", default=PROMPT)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    raw = json.loads(args.config.read_text())
    cfg = Qwen3_5MoeTextConfig(**raw.get("text_config", raw))
    cfg._attn_implementation = "eager"
    n_layers = args.layers or cfg.num_hidden_layers

    ids = Tokenizer.from_file(str(args.tokenizer)).encode(args.prompt, add_special_tokens=False).ids
    T, H = len(ids), cfg.hidden_size
    (args.out / "ids.bin").write_bytes(np.asarray(ids, dtype=np.int32).tobytes())

    g = Gguf(args.gguf)
    dt = torch.bfloat16
    x = g.rows("token_embd.weight", ids).reshape(1, T, H)
    dump(args.out / "embed.bin", x[0])
    x = x.to(dt)

    rope = m.Qwen3_5MoeTextRotaryEmbedding(cfg)
    pos = torch.arange(T).reshape(1, T)
    cos, sin = rope(x, pos)
    mask = torch.full((T, T), torch.finfo(dt).min, dtype=dt).triu(1).reshape(1, 1, T, T)

    with torch.no_grad():
        for i in range(n_layers):
            t0 = time.time()
            layer = m.Qwen3_5MoeDecoderLayer(cfg, i)
            missing, unexpected = layer.load_state_dict(load_layer_state(g, cfg, i), strict=False)
            assert not unexpected, unexpected
            assert not missing, missing
            layer = layer.to(dt).eval()
            res = x
            h = layer.input_layernorm(x)
            if cfg.layer_types[i] == "linear_attention":
                h = layer.linear_attn(h)
            else:
                h, _ = layer.self_attn(hidden_states=h, position_embeddings=(cos, sin), attention_mask=mask)
            x = res + h
            res = x
            hn = layer.post_attention_layernorm(x)
            _, rw, ri = layer.mlp.gate(hn.reshape(-1, H))
            (args.out / f"route{i}.bin").write_bytes(ri.to(torch.int32).numpy().tobytes())
            (args.out / f"routew{i}.bin").write_bytes(rw.to(torch.float32).numpy().tobytes())
            h = layer.mlp(hn)
            if isinstance(h, tuple):
                h = h[0]
            x = res + h
            dump(args.out / f"layer{i}.bin", x[0])
            del layer
            print(f"layer {i} ({cfg.layer_types[i]}) {time.time() - t0:.1f}s", flush=True)

        if n_layers == cfg.num_hidden_layers:
            norm = m.Qwen3_5MoeRMSNorm(H, eps=cfg.rms_norm_eps)
            norm.weight.data = g.get("output_norm.weight") - 1.0
            last = norm.to(dt)(x[0, -1:]).to(torch.float32)
            head = g.get("output.weight")
            logits = (last @ head.T)[0]
            dump(args.out / "logits.bin", logits)
            tok = Tokenizer.from_file(str(args.tokenizer))
            top = torch.topk(logits, 5)
            print("top5:", [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)])

    (args.out / "meta.txt").write_text(f"{T} {H} {n_layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
