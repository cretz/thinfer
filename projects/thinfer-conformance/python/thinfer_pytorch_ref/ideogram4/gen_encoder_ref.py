"""Ideogram-4 text-encoder parity reference.

Loads the Qwen3-VL-8B language model FROM THE SAME GGUF the engine loads
(dequantized Q8_0 -> bf16, so parity isolates kernel correctness from
quantization loss, per `ideogram-plan.md` "QUANT DECISION"), runs the prompt
through the 36 decoder layers, captures the 13 tapped layer outputs
(`QWEN3_VL_ACTIVATION_LAYERS = [0,3,..,33,35]`, the output AFTER each tapped
layer, pre-final-norm), and dumps:

  token_ids.bin            u32 LE, the chat-templated prompt tokens (no special
                           tokens added on top). The engine reads these so both
                           sides tokenize identically (the Rust side owns no
                           Qwen3-VL chat template).
  py_ideo_tap{j}.bin       j in 0..12. The j-th tapped layer output, [seq, 4096]
                           f32, in tap order.
  py_ideo_feats.bin        stacked features [seq, 4096*13 = 53248] f32, feature
                           index for (t, h, j) = t*53248 + h*13 + j (upstream
                           permute(B,L,H,n_taps).reshape(B,L,H*n_taps)).

Memory: builds the 8B text model in bf16 (~16GB) and copies each dequantized
GGUF tensor in-place (peak ~ model + one transient fp32 tensor, ~18GB). Run as
its own process so it frees on exit (never co-resident with the DiT/VAE refs).

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.ideogram4.gen_encoder_ref \\
        --gguf <Qwen3-VL-8B-Instruct-Q8_0.gguf> --prompt "..." --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = "Qwen/Qwen3-VL-8B-Instruct"
# QWEN3_VL_ACTIVATION_LAYERS from third-party/ideogram4 constants.py.
TAP_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(reader_tensor) -> np.ndarray:
    """Flat fp32 of one GGUF tensor (Q8_0 / F32 / F16)."""
    import gguf

    deq = None
    for fn in ("dequantize",):
        f = getattr(gguf, fn, None)
        if f is not None:
            deq = f(reader_tensor.data, reader_tensor.tensor_type)
            break
    if deq is None:  # older gguf
        deq = gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type)
    return np.ascontiguousarray(deq).astype(np.float32).reshape(-1)


def _gguf_to_hf_key(name: str) -> str | None:
    """Map a GGUF tensor name to a Qwen3VLTextModel state-dict key, or None to
    skip (lm head / unused). Mirrors z_image::qwen3_gguf_renames but without the
    `model.` prefix (the bare text model has none)."""
    sites = {
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name == "output.weight":
        return None  # lm head; the text model has none
    if name.startswith("blk."):
        _, idx, rest = name.split(".", 2)
        site = rest.rsplit(".", 1)[0]  # drop trailing `.weight`
        hf = sites.get(site)
        if hf is None:
            raise ValueError(f"unmapped GGUF site {site!r} in {name!r}")
        return f"layers.{idx}.{hf}.weight"
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
            raise ValueError(
                f"{t.name} -> {key}: {flat.size} elems vs param {param.numel()}"
            )
        # ggml stores in-features contiguous (ne0 fastest); HF [out, in] also
        # has in-features last -> flat orders match, plain reshape is correct.
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(key)
        del flat
    missing = set(state) - seen
    # rotary_emb has no parameters; everything else must be loaded.
    missing = {k for k in missing if "rotary" not in k}
    if missing:
        raise RuntimeError(f"GGUF did not cover {len(missing)} params: {sorted(missing)[:5]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, Qwen3VLConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLTextRotaryEmbedding,
    )

    # --- tokenize (chat template, no extra special tokens) ---
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]  # [1, L]
    seq = int(input_ids.shape[1])
    input_ids.to(torch.int32).numpy().astype("<u4").tofile(str(args.out / "token_ids.bin"))
    print(f"[py-ideo-enc] prompt -> {seq} tokens", flush=True)

    # --- build the text tower + load the SAME GGUF the engine loads ---
    cfg = Qwen3VLConfig.from_pretrained(REPO)
    text_cfg = cfg.get_text_config()
    torch.set_grad_enabled(False)
    # Construct in bf16 (~16GB) to avoid a 32GB fp32 spike. The rotary buffer
    # would then be bf16 (truncated frequencies); rebuild it fresh in fp32.
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3VLTextModel(text_cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(text_cfg)  # fp32 inv_freq
    print("[py-ideo-enc] loading GGUF weights (dequant Q8_0 -> bf16)...", flush=True)
    _load_gguf_into(model, args.gguf)

    # --- manual layer loop, byte-faithful to pipeline_ideogram4._get_qwen3_vl_embeddings ---
    inputs_embeds = model.embed_tokens(input_ids)
    pos = torch.arange(seq).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids = pos[0]
    mrope_position_ids = pos[1:]
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(TAP_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    taps = [captured[i][0] for i in TAP_LAYERS]  # each [seq, H]
    for j, tap in enumerate(taps):
        _dump(tap, args.out / f"py_ideo_tap{j}.bin")

    # stack -> (n_taps, seq, H) -> permute (seq, H, n_taps) -> reshape (seq, H*n_taps)
    stacked = torch.stack(taps, dim=0).permute(1, 2, 0).reshape(seq, -1)
    _dump(stacked, args.out / "py_ideo_feats.bin")
    print(f"[py-ideo-enc] done: {len(taps)} taps, feats {tuple(stacked.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
