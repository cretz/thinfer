"""Gemma 4 MoE (`gemma4`, text) reference: the REAL transformers modules
(`Gemma4TextDecoderLayer`'s attention, MLP, router, experts and norms) over the
SAME GGUF the engine reads, one decoder layer resident at a time in bf16 (peak
RAM ~ one layer), 2 torch threads.

GGUF -> HF names (llama.cpp's Gemma 4 converter keeps norms plain):
  * `ffn_gate_up_exps` [E, 2 ff, H] = `experts.gate_up_proj` (gate rows first),
  * `ffn_gate_inp.scale` = `router.scale`, `ffn_down_exps.scale` =
    `router.per_expert_scale`, `layer_output_scale` = `layer_scalar`,
  * `pre_ffw_norm_2` / `post_ffw_norm_{1,2}` / `post_ffw_norm` = the feed-forward
    norms, tied LM head = `token_embd`.

Outputs (all f32 little-endian, row-major):
  ids.bin        i32 [T]            token ids of the prompt (BOS first).
  embed.bin      f32 [T, H]         scaled embedding gather.
  layer{i}.bin   f32 [T, H]         residual stream after decoder layer i.
  route{i}.bin   i32 [T, k]         router top-k expert ids of layer i.
  routew{i}.bin  f32 [T, k]         their routing weights.
  logits.bin     f32 [V]            last-row soft-capped logits (all layers ran).
  meta.txt       "T H layers"

Usage:
  uv run --with gguf --with "transformers>=5.8" python -m thinfer_pytorch_ref.llm.gemma4_ref \
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

from thinfer_pytorch_ref.llm.qwen35moe_ref import Gguf, dump

PROMPT = "The capital of France is Paris. The capital of Germany is"
BOS = 2


def load_layer_state(g: Gguf, sliding: bool, i: int) -> dict[str, torch.Tensor]:
    """GGUF blk.{i} -> HF `model.layers.{i}` state dict (decoder-layer-relative names)."""
    b = f"blk.{i}."
    sd: dict[str, torch.Tensor] = {
        "input_layernorm.weight": g.get(b + "attn_norm.weight"),
        "post_attention_layernorm.weight": g.get(b + "post_attention_norm.weight"),
        "pre_feedforward_layernorm.weight": g.get(b + "ffn_norm.weight"),
        "post_feedforward_layernorm.weight": g.get(b + "post_ffw_norm.weight"),
        "post_feedforward_layernorm_1.weight": g.get(b + "post_ffw_norm_1.weight"),
        "pre_feedforward_layernorm_2.weight": g.get(b + "pre_ffw_norm_2.weight"),
        "post_feedforward_layernorm_2.weight": g.get(b + "post_ffw_norm_2.weight"),
        "layer_scalar": g.get(b + "layer_output_scale.weight").reshape(1),
        "self_attn.q_proj.weight": g.get(b + "attn_q.weight"),
        "self_attn.k_proj.weight": g.get(b + "attn_k.weight"),
        "self_attn.o_proj.weight": g.get(b + "attn_output.weight"),
        "self_attn.q_norm.weight": g.get(b + "attn_q_norm.weight"),
        "self_attn.k_norm.weight": g.get(b + "attn_k_norm.weight"),
        "mlp.gate_proj.weight": g.get(b + "ffn_gate.weight"),
        "mlp.up_proj.weight": g.get(b + "ffn_up.weight"),
        "mlp.down_proj.weight": g.get(b + "ffn_down.weight"),
        "router.proj.weight": g.get(b + "ffn_gate_inp.weight"),
        "router.scale": g.get(b + "ffn_gate_inp.scale"),
        "router.per_expert_scale": g.get(b + "ffn_down_exps.scale"),
        "experts.gate_up_proj": g.get(b + "ffn_gate_up_exps.weight"),
        "experts.down_proj": g.get(b + "ffn_down_exps.weight"),
    }
    if sliding:
        sd["self_attn.v_proj.weight"] = g.get(b + "attn_v.weight")
    return sd


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
    from transformers.models.gemma4 import modeling_gemma4 as m
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    raw = json.loads(args.config.read_text())
    cfg = Gemma4TextConfig(**raw.get("text_config", raw))
    cfg._attn_implementation = "eager"
    n_layers = args.layers or cfg.num_hidden_layers

    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = [BOS] + tok.encode(args.prompt, add_special_tokens=False).ids
    T, H = len(ids), cfg.hidden_size
    (args.out / "ids.bin").write_bytes(np.asarray(ids, dtype=np.int32).tobytes())

    g = Gguf(args.gguf)
    dt = torch.bfloat16
    # Gemma4TextScaledWordEmbedding: rows * sqrt(H), the scale cast to bf16.
    scale = torch.tensor(H**0.5).to(dt)
    x = (g.rows("token_embd.weight", ids).to(dt) * scale).reshape(1, T, H)
    dump(args.out / "embed.bin", x[0])

    rope = m.Gemma4TextRotaryEmbedding(cfg)
    pos = torch.arange(T).reshape(1, T)
    mask = torch.full((T, T), torch.finfo(dt).min, dtype=dt).triu(1).reshape(1, 1, T, T)
    assert T <= cfg.sliding_window, "prompt longer than the sliding window: mask it"

    with torch.no_grad():
        for i in range(n_layers):
            t0 = time.time()
            kind = cfg.layer_types[i]
            sliding = kind == "sliding_attention"
            layer = m.Gemma4TextDecoderLayer(cfg, i)
            missing, unexpected = layer.load_state_dict(load_layer_state(g, sliding, i), strict=False)
            assert not unexpected, unexpected
            assert not missing, missing
            layer = layer.to(dt).eval()
            cos, sin = rope(x, pos, kind)
            # Gemma4TextDecoderLayer.forward, with the router outputs kept.
            res = x
            h = layer.input_layernorm(x)
            h, _ = layer.self_attn(
                hidden_states=h,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                shared_kv_states={},
            )
            x = res + layer.post_attention_layernorm(h)
            res = x
            h1 = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(x)))
            flat = res.reshape(-1, H)
            _, rw, ri = layer.router(flat)
            (args.out / f"route{i}.bin").write_bytes(ri.to(torch.int32).numpy().tobytes())
            (args.out / f"routew{i}.bin").write_bytes(rw.to(torch.float32).numpy().tobytes())
            h2 = layer.experts(layer.pre_feedforward_layernorm_2(flat), ri, rw).reshape(res.shape)
            h = h1 + layer.post_feedforward_layernorm_2(h2)
            x = (res + layer.post_feedforward_layernorm(h)) * layer.layer_scalar
            dump(args.out / f"layer{i}.bin", x[0])
            del layer
            print(f"layer {i} ({kind}) {time.time() - t0:.1f}s", flush=True)

        if n_layers == cfg.num_hidden_layers:
            norm = m.Gemma4RMSNorm(H, eps=cfg.rms_norm_eps)
            norm.weight.data = g.get("output_norm.weight")
            last = norm.to(dt)(x[0, -1:]).to(torch.float32)
            head = g.get("token_embd.weight")
            logits = (last @ head.T)[0]
            cap = cfg.final_logit_softcapping
            if cap:
                logits = torch.tanh(logits / cap) * cap
            dump(args.out / "logits.bin", logits)
            top = torch.topk(logits, 5)
            print("top5:", [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)])

    (args.out / "meta.txt").write_text(f"{T} {H} {n_layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
