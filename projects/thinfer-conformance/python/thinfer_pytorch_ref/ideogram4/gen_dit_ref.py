"""Ideogram-4 DiT parity reference.

Loads the `Ideogram4Transformer` FROM THE SAME DiT GGUF the engine loads
(dequantized Q8_0 -> bf16, isolating kernel correctness from quant loss, per
`ideogram-plan.md`), generates deterministic `llm_features` (text rows) + noise
(image rows), runs ONE velocity forward over the packed `[text][image]`
sequence, and dumps:

  llm_features.bin   f32 LE [num_text, 53248]  the encoder-tap features (text).
  noise.bin          f32 LE [num_image, 128]   noise at image positions.
  velocity.bin       f32 LE [num_image, 128]   the velocity at image positions.
  adaln_input.bin    f32 LE [512]              shared modulation conditioning.
  block0_out.bin     f32 LE [seq, 4608]        h after transformer block 0.
  block_last_out.bin f32 LE [seq, 4608]        h after the last block.
  meta.txt           "num_text num_image grid_h grid_w timestep".

The engine reads llm_features/noise so both sides use identical inputs. No
left-pad: the packed sequence is [text][image] with all segment_ids=1 (the
upstream max_text_tokens pad rows are segment -1 and never influence the image
outputs we read back).

Memory: the DiT is ~9B params; built bf16 (~18GB) with in-place dequant copy
(peak ~ model + one fp32 transient tensor). Run as its own process.

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.ideogram4.gen_dit_ref \\
        --gguf <ideogram4-Q8_0.gguf> --out <dir> \\
        --width 64 --height 64 --num-text 8 --timestep 0.5
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Ground-truth modeling code lives in the third-party clone. Register a stub
# `ideogram4` package (pointing __path__ at the source dir) so submodule imports
# (`ideogram4.constants`, `ideogram4.modeling_ideogram4`) resolve WITHOUT running
# the real heavy __init__ (which drags in diffusers / pipeline / safety).
def _find_third_party() -> Path:
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "third-party" / "ideogram4" / "src"
        if cand.exists():
            return cand
    raise FileNotFoundError("third-party/ideogram4/src not found above this file")


_THIRD_PARTY = _find_third_party()
IMAGE_POSITION_OFFSET = 65536
LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2
PATCH = 16  # patch_size(2) * ae_scale_factor(8)


def _import_modeling():
    src = _THIRD_PARTY
    if not (src / "ideogram4" / "modeling_ideogram4.py").exists():
        raise FileNotFoundError(f"third-party ideogram4 not found at {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "ideogram4" not in sys.modules:
        pkg = types.ModuleType("ideogram4")
        pkg.__path__ = [str(src / "ideogram4")]
        sys.modules["ideogram4"] = pkg
    from ideogram4 import modeling_ideogram4 as m  # noqa: E402

    return m


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(reader_tensor) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    if deq is not None:
        arr = deq(reader_tensor.data, reader_tensor.tensor_type)
    else:
        arr = gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type)
    return np.ascontiguousarray(arr).astype(np.float32).reshape(-1)


def _load_gguf_into(model: torch.nn.Module, gguf_path: Path) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()
    for t in reader.tensors:
        # DiT GGUF keys are 1:1 with module paths (no rename map).
        key = t.name
        param = state.get(key)
        if param is None:
            raise KeyError(f"GGUF tensor {t.name!r} not in model state dict")
        flat = _dequantize(t)
        if flat.size != param.numel():
            raise ValueError(f"{key}: {flat.size} elems vs param {param.numel()}")
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(key)
        del flat
    missing = set(state) - seen
    if missing:
        raise RuntimeError(f"GGUF did not cover {len(missing)} params: {sorted(missing)[:8]}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--num-text", type=int, default=8)
    p.add_argument("--timestep", type=float, default=0.5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assert args.width % PATCH == 0 and args.height % PATCH == 0, "dims must be /16"
    grid_h = args.height // PATCH
    grid_w = args.width // PATCH
    num_image = grid_h * grid_w
    num_text = args.num_text
    seq = num_text + num_image

    m = _import_modeling()
    cfg = m.Ideogram4Config()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    # Deterministic inputs (f32; the model casts to param dtype internally,
    # exactly as the engine uploads the f32 .bin as bf16).
    llm_features_text = torch.randn(num_text, cfg.llm_features_dim, dtype=torch.float32)
    noise_image = torch.randn(num_image, cfg.in_channels, dtype=torch.float32)
    _dump(llm_features_text, args.out / "llm_features.bin")
    _dump(noise_image, args.out / "noise.bin")

    # Packed [text][image]: x is noise at image rows (zero text rows); features
    # are real at text rows (zero image rows).
    x = torch.zeros(1, seq, cfg.in_channels, dtype=torch.float32)
    x[0, num_text:, :] = noise_image
    feats = torch.zeros(1, seq, cfg.llm_features_dim, dtype=torch.float32)
    feats[0, :num_text, :] = llm_features_text

    indicator = torch.zeros(1, seq, dtype=torch.long)
    indicator[0, :num_text] = LLM_TOKEN_INDICATOR
    indicator[0, num_text:] = OUTPUT_IMAGE_INDICATOR
    segment_ids = torch.ones(1, seq, dtype=torch.long)

    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    for pi in range(num_text):
        position_ids[0, pi, :] = pi
    for r in range(grid_h):
        for c in range(grid_w):
            row = num_text + r * grid_w + c
            position_ids[0, row, 0] = IMAGE_POSITION_OFFSET
            position_ids[0, row, 1] = r + IMAGE_POSITION_OFFSET
            position_ids[0, row, 2] = c + IMAGE_POSITION_OFFSET

    t = torch.full((1,), float(args.timestep), dtype=torch.float32)

    torch.set_default_dtype(torch.bfloat16)
    try:
        model = m.Ideogram4Transformer(cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    print("[py-ideo-dit] loading GGUF (dequant Q8_0 -> bf16)...", flush=True)
    _load_gguf_into(model, args.gguf)

    # adaln_input tap: silu(adaln_proj(t_embedding(t))).
    t_cond = model.t_embedding(t)
    if t.dim() == 1:
        t_cond = t_cond.unsqueeze(1)
    adaln_input = torch.nn.functional.silu(model.adaln_proj(t_cond))
    _dump(adaln_input[0, 0], args.out / "adaln_input.bin")

    # Block taps via forward hooks (output h of block 0 and the last block).
    caps: dict[int, torch.Tensor] = {}
    last = len(model.layers) - 1
    h0 = model.layers[0].register_forward_hook(lambda mod, i, o: caps.__setitem__(0, o))
    hl = model.layers[last].register_forward_hook(lambda mod, i, o: caps.__setitem__(last, o))

    out = model(
        llm_features=feats,
        x=x,
        t=t,
        position_ids=position_ids,
        segment_ids=segment_ids,
        indicator=indicator,
    )
    h0.remove()
    hl.remove()

    _dump(caps[0][0], args.out / "block0_out.bin")
    _dump(caps[last][0], args.out / "block_last_out.bin")
    velocity = out[0, num_text:, :]
    _dump(velocity, args.out / "velocity.bin")
    (args.out / "meta.txt").write_text(
        f"{num_text} {num_image} {grid_h} {grid_w} {args.timestep}\n"
    )
    print(
        f"[py-ideo-dit] done: seq={seq} num_image={num_image} velocity{tuple(velocity.shape)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
