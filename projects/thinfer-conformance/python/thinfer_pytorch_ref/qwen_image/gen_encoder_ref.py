"""Qwen-Image text-encoder parity reference.

Loads the Qwen2.5-VL-7B language model FROM THE SAME GGUF the engine loads
(dequantized Q8_0 -> bf16, so parity isolates kernel correctness from quant
loss), runs the (chat-templated) prompt through the 28 decoder layers + final
norm, and dumps `hidden_states[-1]` = `encoder.hidden_states[-1]` (the surface
the Qwen-Image DiT conditions on), plus the token ids so both sides tokenize
identically.

  token_ids.bin   u32 LE, the chat-templated prompt tokens.
  hidden.bin      f32 LE [seq, 3584], the last layer output AFTER `output_norm`.
  meta.txt        "seq hidden"

Text-only MRoPE: position_ids are identical across the 3 (t,h,w) axes, so the
mrope_section [16,24,24] split reduces to standard 1-axis rotary (what the
engine runs). Memory: 7B bf16 (~15GB); run as its own process so it frees on
exit.

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.qwen_image.gen_encoder_ref \\
        --gguf <Qwen2.5-VL-7B...Q8_0.gguf> --tokenizer-dir <dir> \\
        --prompt "..." --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Qwen-Image t2i prompt template (pipeline_qwenimage.py); drop_idx is a pipeline
# concern (applied at e2e), so here we encode + compare the full token sequence.
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

HIDDEN = 3584
N_LAYERS = 28
N_HEADS = 28
N_KV_HEADS = 4
HEAD_DIM = 128
FFN = 18944
VOCAB = 152064
EPS = 1e-6
ROPE_THETA = 1_000_000.0
MROPE_SECTION = [16, 24, 24]


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(reader_tensor) -> np.ndarray:
    """Flat fp32 of one GGUF tensor (Q8_0 / F32 / F16)."""
    import gguf

    deq = getattr(gguf, "dequantize", None)
    if deq is not None:
        out = deq(reader_tensor.data, reader_tensor.tensor_type)
    else:  # older gguf
        out = gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type)
    return np.ascontiguousarray(out).astype(np.float32).reshape(-1)


def _gguf_to_hf_key(name: str) -> str | None:
    """GGUF (`qwen2vl` native) -> `Qwen2_5_VLTextModel` state-dict key (bare, no
    `model.` prefix). Mirrors `qwen_image::text_encoder::qwen2vl_gguf_renames`.
    Returns None for the (unused) lm head."""
    sites = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_q.bias": "self_attn.q_proj.bias",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_k.bias": "self_attn.k_proj.bias",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_v.bias": "self_attn.v_proj.bias",
        "attn_output.weight": "self_attn.o_proj.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name == "output.weight":
        return None
    if name.startswith("blk."):
        _, idx, rest = name.split(".", 2)
        hf = sites.get(rest)
        if hf is None:
            raise ValueError(f"unmapped GGUF site {rest!r} in {name!r}")
        return f"layers.{idx}.{hf}"
    raise ValueError(f"unexpected GGUF tensor {name!r}")


def _load_gguf_into(model: torch.nn.Module, gguf_path: Path) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()
    for t in reader.tensors:
        key = _gguf_to_hf_key(t.name)
        if key is None:
            continue
        param = state.get(key)
        if param is None:
            raise KeyError(f"GGUF tensor {t.name!r} -> {key!r} not in model")
        flat = _dequantize(t)
        if flat.size != param.numel():
            raise ValueError(f"{t.name} -> {key}: {flat.size} vs param {param.numel()}")
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(key)
        del flat
    missing = {k for k in (set(state) - seen) if "rotary" not in k}
    if missing:
        raise RuntimeError(f"GGUF did not cover {len(missing)} params: {sorted(missing)[:5]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLTextModel,
    )

    # --- tokenize (Qwen-Image t2i template) ---
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    text = PROMPT_TEMPLATE.format(args.prompt)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]  # [1, L]
    seq = int(input_ids.shape[1])
    input_ids.to(torch.int32).numpy().astype("<u4").tofile(str(args.out / "token_ids.bin"))
    print(f"[py-qwen-enc] prompt -> {seq} tokens", flush=True)

    # --- build the text tower + load the SAME GGUF ---
    cfg = Qwen2_5_VLTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=FFN,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        tie_word_embeddings=False,
        rope_scaling={"rope_type": "default", "mrope_section": MROPE_SECTION, "rope_theta": ROPE_THETA},
    )
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2_5_VLTextModel(cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    print("[py-qwen-enc] loading GGUF weights (dequant Q8_0 -> bf16)...", flush=True)
    _load_gguf_into(model, args.gguf)

    # --- forward: text positions identical across the 3 mrope axes ---
    pos = torch.arange(seq).view(1, 1, seq).expand(3, 1, seq)  # [3, B, seq]
    out = model(
        input_ids=input_ids,
        position_ids=pos,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = out.hidden_states[-1][0]  # [seq, HIDDEN], post output_norm
    _dump(hidden, args.out / "hidden.bin")
    (args.out / "meta.txt").write_text(f"{seq} {HIDDEN}\n")
    print(
        f"[py-qwen-enc] done: hidden{tuple(hidden.shape)} "
        f"range[{hidden.min():.3f},{hidden.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
