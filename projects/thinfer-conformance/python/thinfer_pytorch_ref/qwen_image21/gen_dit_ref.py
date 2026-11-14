"""Qwen-Image-2.1 DiT parity reference (truncated depth, SAME GGUF weights).

The 32-layer 7B DiT in bf16 is ~15GB, so this builds an N-layer
`QwenImage21Transformer2DModel` (the third-party diffusers clone; the venv's
release predates 2.1) and loads the embedders + blocks 0..N from the GGUF the
engine loads (dequantized -> bf16). `img_mlp.gate_up` is `[gate_layer; proj]`
fused along the output axis. Deeper is repetition of the same block.

The joint sequence: text, one condition image, text, then the target image
(the VLM layout: each image slot stands for 2x2 latent tokens). The forward is
the uncached, block-causal one; the engine's prefill + per-step KV reuse must
reproduce it. Two timesteps exercise the cache reuse.

  encoder.bin   f32 [enc_rows, 4096]     encoder rows (VLM positions; image slots unused).
  cond.bin      f32 [ch*cw, 64]          condition latent tokens.
  target.bin    f32 [th*tw, 64]          target latent tokens.
  vel_{i}.bin   f32 [th*tw, 64]          target velocity at timesteps[i].
  meta.txt      "layers t0 .. tN" then the layout line
                "text_a ch cw text_b th tw"

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.qwen_image21.gen_dit_ref \\
        --gguf <qwen-image-2.1-turbo-Q8_0.gguf> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PREFIX = "model.diffusion_model."
TEXT_DIM = 4096
LATENT_CH = 64
TOKENS_PER_SLOT = 4


def _locate_diffusers_src() -> str:
    for p in Path(__file__).resolve().parents:
        cand = p / "third-party" / "diffusers" / "src"
        if (cand / "diffusers" / "models" / "transformers" / "transformer_qwenimage21.py").exists():
            return str(cand)
    raise AssertionError("could not locate third-party/diffusers (with Qwen-Image-2.1) above this script")


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(rt) -> np.ndarray:
    import gguf

    return np.ascontiguousarray(gguf.quants.dequantize(rt.data, rt.tensor_type)).astype(np.float32)


def _load_gguf_into(model: torch.nn.Module, gguf_path: Path, layers: int) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()

    def put(name: str, flat: np.ndarray) -> None:
        param = state[name]
        if flat.size != param.numel():
            raise ValueError(f"{name}: {flat.size} vs param {param.numel()}")
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(name)

    for t in reader.tensors:
        name = t.name.removeprefix(PREFIX)
        if name.startswith("transformer_blocks.") and int(name.split(".")[1]) >= layers:
            continue
        w = _dequantize(t).reshape(-1)
        if name.endswith(".img_mlp.gate_up.weight"):
            base = name.removesuffix("gate_up.weight")
            half = w.size // 2
            put(base + "gate_layer.weight", w[:half])
            put(base + "proj.weight", w[half:])
        else:
            put(name, w)
    missing = set(state) - seen
    if missing:
        raise RuntimeError(f"GGUF did not cover {len(missing)} params: {sorted(missing)[:6]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--text-a", type=int, default=6)
    p.add_argument("--cond", type=int, nargs=2, default=(4, 4), metavar=("H", "W"))
    p.add_argument("--text-b", type=int, default=5)
    p.add_argument("--target", type=int, nargs=2, default=(4, 6), metavar=("H", "W"))
    p.add_argument("--timesteps", type=float, nargs="+", default=(0.95418, 0.414568))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, _locate_diffusers_src())
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    (ch, cw), (th, tw) = args.cond, args.target
    assert ch % 2 == 0 and cw % 2 == 0 and th % 2 == 0 and tw % 2 == 0
    cond_slots, target_slots = ch * cw // TOKENS_PER_SLOT, th * tw // TOKENS_PER_SLOT
    enc_rows = args.text_a + cond_slots + args.text_b

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = QwenImage21Transformer2DModel(num_layers=args.layers).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    _load_gguf_into(model, args.gguf, args.layers)

    encoder = torch.randn(enc_rows, TEXT_DIM)
    cond = torch.randn(ch * cw, LATENT_CH)
    target = torch.randn(th * tw, LATENT_CH)
    _dump(encoder, args.out / "encoder.bin")
    _dump(cond, args.out / "cond.bin")
    _dump(target, args.out / "target.bin")

    img_mask = torch.zeros(1, enc_rows + target_slots, dtype=torch.bool)
    img_mask[0, args.text_a : args.text_a + cond_slots] = True
    img_mask[0, enc_rows:] = True
    for i, t in enumerate(args.timesteps):
        out = model(
            hidden_states=torch.cat([cond, target])[None].to(torch.bfloat16),
            encoder_hidden_states=encoder[None].to(torch.bfloat16),
            timestep=torch.tensor([t]),
            img_shapes=[[(1, ch, cw), (1, th, tw)]],
            img_mask=img_mask,
            return_dict=True,
        ).sample[0]
        vel = out[-th * tw :]
        _dump(vel, args.out / f"vel_{i}.bin")
        print(f"[py-qwen21-dit] t={t} vel{tuple(vel.shape)} range[{vel.min():.3f},{vel.max():.3f}]", flush=True)

    ts = " ".join(str(t) for t in args.timesteps)
    (args.out / "meta.txt").write_text(
        f"{args.layers} {ts}\n{args.text_a} {ch} {cw} {args.text_b} {th} {tw}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
