"""Qwen3-VL-32B vision tower reference (MiniMax-H3's image conditioner): HF
`Qwen3VLVisionModel` in bf16 on CPU over the SAME mmproj GGUF the engine reads
(weights mapped from llama.cpp names), on a deterministic synthetic image.

The processor math is reproduced in numpy (the fast image processor needs
torchvision): `(u8 - 127.5) / 127.5`, patches merge-unit-major, each row
`(C, T=2, 16, 16)` with the frame duplicated.

Dumps (little-endian): `image.bin` (u8 `[H, W, 3]`), `embeds.bin` (f32
`[tokens, 5120]`), `deepstack{0,1,2}.bin` (same shape), `meta.txt` (`H W`).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30


def read_gguf(path: Path) -> dict[str, np.ndarray]:
    """Minimal GGUF reader for an all-float file (F32/F16/BF16 tensors)."""
    data = path.read_bytes()
    off = 0

    def take(fmt: str):
        nonlocal off
        vals = struct.unpack_from("<" + fmt, data, off)
        off += struct.calcsize("<" + fmt)
        return vals

    def string() -> str:
        nonlocal off
        (n,) = take("Q")
        s = data[off : off + n].decode()
        off += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t: int):
        if t == 8:
            return string()
        if t == 9:
            at, n = take("IQ")
            return [value(at) for _ in range(n)]
        return take(scalar[t])[0]

    assert data[:4] == b"GGUF"
    off = 4
    take("I")
    n_tensors, n_kv = take("QQ")
    align = 32
    for _ in range(n_kv):
        key = string()
        (t,) = take("I")
        v = value(t)
        if key == "general.alignment":
            align = v
    infos = []
    for _ in range(n_tensors):
        name = string()
        (nd,) = take("I")
        dims = take("Q" * nd)
        (ty,) = take("I")
        (toff,) = take("Q")
        infos.append((name, dims, ty, toff))
    base = (off + align - 1) // align * align
    out = {}
    for name, dims, ty, toff in infos:
        shape = tuple(reversed(dims))
        n = int(np.prod(shape))
        start = base + toff
        if ty == GGML_F32:
            arr = np.frombuffer(data, np.float32, n, start)
        elif ty == GGML_F16:
            arr = np.frombuffer(data, np.float16, n, start).astype(np.float32)
        elif ty == GGML_BF16:
            raw = np.frombuffer(data, np.uint16, n, start).astype(np.uint32) << 16
            arr = raw.view(np.float32)
        else:
            raise ValueError(f"{name}: unsupported ggml type {ty}")
        out[name] = arr.reshape(shape)
    return out


def state_dict(g: dict[str, np.ndarray], depth: int, taps: list[int]) -> dict[str, torch.Tensor]:
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    sd = {
        # Conv3d [out, C, T, P, P]: the two temporal slabs.
        "patch_embed.proj.weight": t(np.stack([g["v.patch_embd.weight"], g["v.patch_embd.weight.1"]], axis=2)),
        "patch_embed.proj.bias": t(g["v.patch_embd.bias"]),
        "pos_embed.weight": t(g["v.position_embd.weight"]),
        "merger.norm.weight": t(g["v.post_ln.weight"]),
        "merger.norm.bias": t(g["v.post_ln.bias"]),
        "merger.linear_fc1.weight": t(g["mm.0.weight"]),
        "merger.linear_fc1.bias": t(g["mm.0.bias"]),
        "merger.linear_fc2.weight": t(g["mm.2.weight"]),
        "merger.linear_fc2.bias": t(g["mm.2.bias"]),
    }
    names = {
        "norm1": "ln1",
        "norm2": "ln2",
        "attn.qkv": "attn_qkv",
        "attn.proj": "attn_out",
        "mlp.linear_fc1": "ffn_up",
        "mlp.linear_fc2": "ffn_down",
    }
    for i in range(depth):
        for hf, gg in names.items():
            for p in ("weight", "bias"):
                sd[f"blocks.{i}.{hf}.{p}"] = t(g[f"v.blk.{i}.{gg}.{p}"])
    for k, layer in enumerate(taps):
        for hf, gg in (("norm", "norm"), ("linear_fc1", "fc1"), ("linear_fc2", "fc2")):
            for p in ("weight", "bias"):
                sd[f"deepstack_merger_list.{k}.{hf}.{p}"] = t(g[f"v.deepstack.{layer}.{gg}.{p}"])
    return sd


def synthetic_image(h: int, w: int) -> np.ndarray:
    """Smooth gradients plus seeded noise: structure at every scale."""
    rng = np.random.default_rng(1234)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack(
        [
            127 + 100 * np.sin(x / 37.0) * np.cos(y / 53.0),
            127 + 100 * np.cos((x + y) / 71.0),
            (x / w) * 200 + 20,
        ],
        axis=-1,
    )
    img += rng.normal(0, 12, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def patchify(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    h, w, _ = img.shape
    x = (img.astype(np.float32) - 127.5) / 127.5
    x = x.transpose(2, 0, 1)[None]  # [1, C, H, W]
    gh, gw = h // 16, w // 16
    x = x.reshape(1, 3, gh // 2, 2, 16, gw // 2, 2, 16).transpose(0, 2, 5, 3, 6, 1, 4, 7)
    x = np.repeat(x[:, :, :, :, :, :, None], 2, axis=6)  # T = 2, duplicated
    return x.reshape(gh * gw, 3 * 2 * 16 * 16), (1, gh, gw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(out_hidden_size=5120)
    cfg._attn_implementation = "eager"
    model = Qwen3VLVisionModel(cfg)
    g = read_gguf(args.mmproj)
    missing, unexpected = model.load_state_dict(
        state_dict(g, cfg.depth, list(cfg.deepstack_visual_indexes)), strict=True
    )
    assert not missing and not unexpected, (missing, unexpected)
    del g
    model = model.bfloat16().eval()

    img = synthetic_image(args.height, args.width)
    pixels, thw = patchify(img)
    out = model(
        torch.from_numpy(pixels).bfloat16(),
        grid_thw=torch.tensor([thw]),
    )
    (args.out / "image.bin").write_bytes(img.tobytes())
    (args.out / "embeds.bin").write_bytes(out.pooler_output.float().numpy().astype("<f4").tobytes())
    for k, ds in enumerate(out.deepstack_features):
        (args.out / f"deepstack{k}.bin").write_bytes(ds.float().numpy().astype("<f4").tobytes())
    (args.out / "meta.txt").write_text(f"{args.height} {args.width}\n")
    print("tokens", out.pooler_output.shape, "deepstack", len(out.deepstack_features))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
