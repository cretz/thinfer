"""Qwen-Image DiT single-block parity reference.

The full 60-layer 20B DiT in bf16 is ~41GB (won't fit 64GB w/ headroom), so this
builds a 1-LAYER `QwenImageTransformer2DModel` and loads ONLY block 0 + the
embedders from the SAME GGUF the engine loads (dequant Q8_0 -> bf16). 60-layer
depth is just repetition of this exact block; this validates every kernel
(img_in/txt_in/txt_norm/time_embed, the dual-stream block, norm_out/proj_out,
complex RoPE, joint attention) against the diffusers ground truth.

Seeds img latent tokens + text embeds, runs the 1-block forward, dumps:

  img_tokens.bin  f32 [img_seq, 64]     packed latent tokens (DiT input).
  txt_embeds.bin  f32 [txt_seq, 3584]   encoder hidden states (DiT text input).
  temb.bin        f32 [3072]            time_text_embed output.
  block0_hs.bin   f32 [img_seq, 3072]   image stream after block 0.
  block0_eh.bin   f32 [txt_seq, 3072]   text stream after block 0.
  velocity.bin    f32 [img_seq, 64]     proj_out(norm_out(hs, temb)).
  meta.txt        "img_seq txt_seq gh gw"

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.qwen_image.gen_dit_ref \\
        --gguf <qwen-rapid-...-Q8_0.gguf> --out <dir> --width 64 --height 64 \\
        --txt-seq 8 --timestep 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PIXELS_PER_TOKEN = 16  # vae_scale_factor(8) * patch(2)
IN_CH = 64
JOINT_DIM = 3584


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(rt) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    out = deq(rt.data, rt.tensor_type) if deq is not None else gguf.quants.dequantize(rt.data, rt.tensor_type)
    return np.ascontiguousarray(out).astype(np.float32).reshape(-1)


def _wanted(name: str) -> bool:
    """Block-0 + embedder tensors only (the 1-layer model). diffusers names are
    1:1 with the GGUF, so no rename."""
    if name.startswith("transformer_blocks."):
        return name.startswith("transformer_blocks.0.")
    return True  # img_in/txt_in/txt_norm/time_text_embed/norm_out/proj_out


def _load_gguf_into(model: torch.nn.Module, gguf_path: Path) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()
    for t in reader.tensors:
        if not _wanted(t.name):
            continue
        param = state.get(t.name)
        if param is None:
            raise KeyError(f"GGUF tensor {t.name!r} not in 1-layer model")
        flat = _dequantize(t)
        if flat.size != param.numel():
            raise ValueError(f"{t.name}: {flat.size} vs param {param.numel()}")
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(t.name)
        del flat
    missing = {k for k in (set(state) - seen) if "pos_embed" not in k}
    if missing:
        raise RuntimeError(f"GGUF did not cover {len(missing)} params: {sorted(missing)[:6]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--txt-seq", type=int, default=8)
    p.add_argument("--timestep", type=float, default=500.0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from diffusers import QwenImageTransformer2DModel

    assert args.width % PIXELS_PER_TOKEN == 0 and args.height % PIXELS_PER_TOKEN == 0
    gh = args.height // PIXELS_PER_TOKEN
    gw = args.width // PIXELS_PER_TOKEN
    img_seq = gh * gw
    txt_seq = args.txt_seq

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=IN_CH,
            out_channels=16,
            num_layers=1,
            attention_head_dim=128,
            num_attention_heads=24,
            joint_attention_dim=JOINT_DIM,
            guidance_embeds=False,
            axes_dims_rope=(16, 56, 56),
        ).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    _load_gguf_into(model, args.gguf)

    # seeded inputs (fp32 -> bf16 for the run)
    img_tokens = torch.randn(1, img_seq, IN_CH, dtype=torch.float32)
    txt_embeds = torch.randn(1, txt_seq, JOINT_DIM, dtype=torch.float32)
    _dump(img_tokens[0], args.out / "img_tokens.bin")
    _dump(txt_embeds[0], args.out / "txt_embeds.bin")

    cap: dict[str, torch.Tensor] = {}
    model.time_text_embed.register_forward_hook(lambda m, i, o: cap.__setitem__("temb", o.detach()))

    def blk_hook(m, i, o):
        cap["eh"], cap["hs"] = o[0].detach(), o[1].detach()

    model.transformer_blocks[0].register_forward_hook(blk_hook)

    out = model(
        hidden_states=img_tokens.to(torch.bfloat16),
        encoder_hidden_states=txt_embeds.to(torch.bfloat16),
        encoder_hidden_states_mask=torch.ones(1, txt_seq, dtype=torch.long),
        timestep=torch.tensor([args.timestep], dtype=torch.float32),
        img_shapes=[(1, gh, gw)],
        return_dict=True,
    ).sample  # [1, img_seq, 64]

    _dump(cap["temb"][0], args.out / "temb.bin")
    _dump(cap["hs"][0], args.out / "block0_hs.bin")
    _dump(cap["eh"][0], args.out / "block0_eh.bin")
    _dump(out[0], args.out / "velocity.bin")
    (args.out / "meta.txt").write_text(f"{img_seq} {txt_seq} {gh} {gw}\n")
    print(
        f"[py-qwen-dit] done: img_seq={img_seq} txt_seq={txt_seq} grid={gh}x{gw} "
        f"vel{tuple(out[0].shape)} range[{out.min():.3f},{out.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
