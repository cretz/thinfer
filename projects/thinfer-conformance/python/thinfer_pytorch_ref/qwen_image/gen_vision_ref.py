"""Qwen2.5-VL vision tower (mmproj) parity reference.

Builds the HF `Qwen2_5_VisionTransformerPretrainedModel` with the config the
mmproj GGUF encodes (depth 32, hidden 1280, 16 heads, intermediate 3420,
out_hidden_size 3584, spatial_merge 2, patch 14, temporal 2, window 112,
fullatt {7,15,23,31}, silu), loads weights by DEQUANTIZING the SAME mmproj GGUF
the engine loads and REMAPPING the clip names to the HF vision module names,
then runs a deterministic patchified image and dumps the projected vision
tokens.

The engine consumes the SAME `pixel_values [N, 1176]` tensor (dumped here), so
the comparison is apples-to-apples: both sides see identical patch input and
identical (dequantized) weights, isolating engine kernel correctness.

  pixel_values.bin  f32 [N, 1176]       patchified image (HF processor layout,
                                        merge-unit-major). N = gh*gw.
  vision_embeds.bin f32 [N/4, 3584]     merger output (pooler_output), raster
                                        merged order = LM <|image_pad|> embeds.
  meta.txt          "gh gw N tokens"

Memory: the vision tower is ~0.6B; fp32 CPU at tiny grids is trivial.

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.qwen_image.gen_vision_ref \\
        --mmproj <...mmproj-f16.gguf> --out <dir> --gh 8 --gw 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

HIDDEN = 1280
DEPTH = 32
HEADS = 16
INTERMEDIATE = 3420
OUT_HIDDEN = 3584
PATCH = 14
TEMPORAL = 2
IN_CH = 3
MERGE = 2
PATCH_ELEMS = IN_CH * TEMPORAL * PATCH * PATCH  # 1176


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(rt) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    out = (
        deq(rt.data, rt.tensor_type)
        if deq is not None
        else gguf.quants.dequantize(rt.data, rt.tensor_type)
    )
    return np.ascontiguousarray(out).astype(np.float32)


def _gguf_tensors(mmproj: Path) -> dict[str, np.ndarray]:
    import gguf

    reader = gguf.GGUFReader(str(mmproj))
    out: dict[str, np.ndarray] = {}
    for t in reader.tensors:
        # GGUF shape is innermost-first; reverse to torch/outer-first order.
        shape = tuple(int(d) for d in reversed(t.shape))
        out[t.name] = _dequantize(t).reshape(shape)
    return out


def _build_state_dict(g: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Remap clip `v.*`/`mm.*` names to HF vision module names."""
    sd: dict[str, torch.Tensor] = {}

    def put(name: str, arr: np.ndarray) -> None:
        sd[name] = torch.from_numpy(np.ascontiguousarray(arr)).float()

    # patch_embed.proj is a Conv3d [1280, 3, 2, 14, 14]: stack the two temporal
    # slabs (each [1280, 3, 14, 14] in torch order) along the temporal dim.
    w0 = g["v.patch_embd.weight"].reshape(HIDDEN, IN_CH, PATCH, PATCH)
    w1 = g["v.patch_embd.weight.1"].reshape(HIDDEN, IN_CH, PATCH, PATCH)
    conv = np.stack([w0, w1], axis=2)  # [1280, 3, 2, 14, 14]
    put("patch_embed.proj.weight", conv)

    for i in range(DEPTH):
        v = f"v.blk.{i}"
        b = f"blocks.{i}"
        put(f"{b}.norm1.weight", g[f"{v}.ln1.weight"])
        put(f"{b}.norm2.weight", g[f"{v}.ln2.weight"])
        # fuse q,k,v -> attn.qkv (concat along out dim).
        qkv_w = np.concatenate(
            [g[f"{v}.attn_q.weight"], g[f"{v}.attn_k.weight"], g[f"{v}.attn_v.weight"]],
            axis=0,
        )
        qkv_b = np.concatenate(
            [g[f"{v}.attn_q.bias"], g[f"{v}.attn_k.bias"], g[f"{v}.attn_v.bias"]],
            axis=0,
        )
        put(f"{b}.attn.qkv.weight", qkv_w)
        put(f"{b}.attn.qkv.bias", qkv_b)
        put(f"{b}.attn.proj.weight", g[f"{v}.attn_out.weight"])
        put(f"{b}.attn.proj.bias", g[f"{v}.attn_out.bias"])
        put(f"{b}.mlp.gate_proj.weight", g[f"{v}.ffn_gate.weight"])
        put(f"{b}.mlp.gate_proj.bias", g[f"{v}.ffn_gate.bias"])
        put(f"{b}.mlp.up_proj.weight", g[f"{v}.ffn_up.weight"])
        put(f"{b}.mlp.up_proj.bias", g[f"{v}.ffn_up.bias"])
        put(f"{b}.mlp.down_proj.weight", g[f"{v}.ffn_down.weight"])
        put(f"{b}.mlp.down_proj.bias", g[f"{v}.ffn_down.bias"])

    put("merger.ln_q.weight", g["v.post_ln.weight"])
    put("merger.mlp.0.weight", g["mm.0.weight"])
    put("merger.mlp.0.bias", g["mm.0.bias"])
    put("merger.mlp.2.weight", g["mm.2.weight"])
    put("merger.mlp.2.bias", g["mm.2.bias"])
    return sd


def _patchify(gh: int, gw: int, seed: int = 7) -> np.ndarray:
    """Deterministic [N, 1176] patchified image, replicating EXACTLY the HF
    `Qwen2VLImageProcessorFast` patchify (reshape -> permute (0,2,5,3,6,1,4,7)
    -> unsqueeze temporal expand -> flatten to `C*T*P*P`). Row layout = [C,T,P,P]
    with T a pure repeat of the single frame."""
    rng = np.random.default_rng(seed)
    h, w = gh * PATCH, gw * PATCH
    img = rng.uniform(-1.0, 1.0, (1, IN_CH, h, w)).astype("float32")  # [B=1, C, H, W]
    patches = img.reshape(1, IN_CH, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    # [B, gh/M, gw/M, M, M, C, P, P]
    patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
    # insert temporal at axis 6 and repeat -> [..., C, T, P, P]
    patches = np.repeat(patches[:, :, :, :, :, :, None], TEMPORAL, axis=6)
    flatten = patches.reshape(gh * gw, IN_CH * TEMPORAL * PATCH * PATCH)
    return np.ascontiguousarray(flatten.astype("float32"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mmproj", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gh", type=int, default=8)
    p.add_argument("--gw", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.gh % MERGE == 0 and args.gw % MERGE == 0, "grid must be /2"

    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    torch.set_grad_enabled(False)

    cfg = Qwen2_5_VLVisionConfig(
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        out_hidden_size=OUT_HIDDEN,
        spatial_merge_size=MERGE,
        patch_size=PATCH,
        temporal_patch_size=TEMPORAL,
        in_channels=IN_CH,
        window_size=112,
        fullatt_block_indexes=[7, 15, 23, 31],
        hidden_act="silu",
    )
    model = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval().to(torch.float32)
    model.config._attn_implementation = "eager"

    g = _gguf_tensors(args.mmproj)
    sd = _build_state_dict(g)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing, f"missing params: {sorted(missing)[:8]}"
    assert not unexpected, f"unexpected params: {sorted(unexpected)[:8]}"

    pixel_values = _patchify(args.gh, args.gw)
    n = args.gh * args.gw
    assert pixel_values.shape == (n, PATCH_ELEMS)
    _dump(torch.from_numpy(pixel_values), args.out / "pixel_values.bin")

    grid_thw = torch.tensor([[1, args.gh, args.gw]], dtype=torch.long)
    out = model(
        hidden_states=torch.from_numpy(pixel_values).float(),
        grid_thw=grid_thw,
    )
    embeds = out.pooler_output  # [N/4, 3584] raster merged order
    tokens = n // (MERGE * MERGE)
    assert embeds.shape == (tokens, OUT_HIDDEN), embeds.shape
    _dump(embeds, args.out / "vision_embeds.bin")
    (args.out / "meta.txt").write_text(f"{args.gh} {args.gw} {n} {tokens}\n")
    print(
        f"[py-qwen-vision] done: grid={args.gh}x{args.gw} N={n} tokens={tokens} "
        f"embeds{tuple(embeds.shape)} range[{embeds.min():.3f},{embeds.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
