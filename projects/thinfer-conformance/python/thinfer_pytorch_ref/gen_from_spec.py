"""Generate per-dtype safetensors fixtures from a spec JSON.

Spec is authored in Rust (`thinfer_core::conformance::SpecCase`); each case
carries a `dtypes` list naming the variants to generate. One safetensors
file per dtype, keyed `<case.name>/<input.name>` for inputs and
`<case.name>/out` for the reference output.

Dtypes:
- `fp32`: pure fp32 ref output.
- `bf16w`: fp32 ref output RNE-quantized to bf16 then back to fp32 - matches
  what our shaders emit when `WgslConfig::BF16_QUANT_WRITES` is active
  (every activation-producing store rounded through bf16).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def quantize_for_dtype(t: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "fp32":
        return t
    if dtype_name == "bf16w":
        return t.to(torch.bfloat16).to(torch.float32)
    raise ValueError(f"unknown dtype: {dtype_name}")


def make_input(fill: dict, shape: list[int]) -> torch.Tensor:
    kind = fill["kind"]
    if kind == "linspace":
        n = 1
        for d in shape:
            n *= d
        lo, hi = float(fill["lo"]), float(fill["hi"])
        x = torch.linspace(lo, hi, n, dtype=torch.float32)
        if fill.get("flip", False):
            x = x.flip(0)
        return x.reshape(shape)
    raise ValueError(f"unknown fill kind: {kind}")


def compute_output(op: str, case: dict, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    if op == "add":
        return inputs["a"] + inputs["b"]
    if op == "mul":
        return inputs["a"] * inputs["b"]
    if op == "silu":
        return F.silu(inputs["x"])
    if op == "silu_mul":
        return F.silu(inputs["a"]) * inputs["b"]
    if op == "tanh":
        return torch.tanh(inputs["x"])
    if op == "bcast_affine":
        return inputs["x"] * (inputs["s"] + float(case["bias"]))
    if op == "bcast_fma":
        return inputs["x"] + inputs["s"] * inputs["y"]
    if op == "bcast_add":
        return inputs["x"] + inputs["s"]
    if op == "matmul":
        return inputs["a"] @ inputs["b"]
    if op == "rmsnorm":
        x = inputs["x"]
        w = inputs["w"]
        eps = float(case["eps"])
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return x * rms * w
    if op == "layernorm":
        x = inputs["x"]
        eps = float(case["eps"])
        mean = x.mean(dim=-1, keepdim=True)
        d = x - mean
        inv = torch.rsqrt(d.pow(2).mean(dim=-1, keepdim=True) + eps)
        return d * inv
    if op == "softmax":
        return F.softmax(inputs["x"], dim=-1)
    if op == "sdpa":
        # BSHD on the wire; transpose to BHSD for the matmul math, then back.
        q = inputs["q"].transpose(1, 2)  # [B, H_q,  S_q, D]
        k = inputs["k"].transpose(1, 2)  # [B, H_kv, S_k, D]
        v = inputs["v"].transpose(1, 2)  # [B, H_kv, S_k, D]
        mask = inputs["mask"]  # [B, S_q, S_k] additive per (query, key)
        scale = float(case["scale"])
        h_q, h_kv = q.shape[1], k.shape[1]
        if h_q != h_kv:
            rep = h_q // h_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        s = (q @ k.transpose(-1, -2)) * scale + mask[:, None, :, :]
        a = F.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).contiguous()
    if op == "conv2d":
        return F.conv2d(
            inputs["x"],
            inputs["w"],
            bias=inputs["bias"],
            stride=(int(case["stride_h"]), int(case["stride_w"])),
            padding=(int(case["pad_h"]), int(case["pad_w"])),
        )
    if op == "transpose12":
        x = inputs["x"]
        return x.transpose(1, 2).contiguous()
    if op == "rope":
        x = inputs["x"]  # [rows, heads, dim]
        f = inputs["freqs"]  # [rows, dim]
        rows, heads, dim = x.shape
        pairs = dim // 2
        xc = torch.view_as_complex(x.reshape(rows, heads, pairs, 2).contiguous())
        fc = torch.view_as_complex(f.reshape(rows, pairs, 2).contiguous()).unsqueeze(1)
        return torch.view_as_real(xc * fc).reshape(rows, heads, dim)
    if op == "rope_halfrot":
        # HuggingFace-style RoPE: pair k uses elements (k, k + D/2).
        # Freqs stay in pair-interleaved layout (cos_k at 2k, sin_k at 2k+1)
        # so the same RopeEmbedder output works for both kernels.
        x = inputs["x"]  # [rows, heads, dim]
        f = inputs["freqs"]  # [rows, dim], cos/sin interleaved per pair
        rows, heads, dim = x.shape
        pairs = dim // 2
        cos = f.reshape(rows, pairs, 2)[..., 0]  # [rows, pairs]
        sin = f.reshape(rows, pairs, 2)[..., 1]  # [rows, pairs]
        cos = cos.unsqueeze(1)  # broadcast over heads -> [rows, 1, pairs]
        sin = sin.unsqueeze(1)
        x_re = x[..., :pairs]  # [rows, heads, pairs]
        x_im = x[..., pairs:]
        out_re = x_re * cos - x_im * sin
        out_im = x_re * sin + x_im * cos
        return torch.cat((out_re, out_im), dim=-1).contiguous()
    raise ValueError(f"unknown op: {op}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    spec = json.loads(args.spec.read_text())

    by_dtype: dict[str, dict[str, torch.Tensor]] = {}

    for case in spec["cases"]:
        name = case["name"]
        # Inputs are always fp32 on disk; bf16w only affects the `out` tensor
        # (mirrors shader behavior: storage is fp32, only the activation write
        # is bf16-quantized).
        inputs: dict[str, torch.Tensor] = {}
        for inp in case["inputs"]:
            inputs[inp["name"]] = make_input(inp["fill"], inp["shape"])

        out_fp32 = compute_output(case["op"], case, inputs)

        for dtype_name in case["dtypes"]:
            bucket = by_dtype.setdefault(dtype_name, {})
            for k, v in inputs.items():
                bucket[f"{name}/{k}"] = v.contiguous()
            bucket[f"{name}/out"] = quantize_for_dtype(out_fp32, dtype_name).contiguous()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dtype_name, tensors in by_dtype.items():
        save_file(tensors, str(args.out_dir / f"{dtype_name}.safetensors"))


if __name__ == "__main__":
    main()
