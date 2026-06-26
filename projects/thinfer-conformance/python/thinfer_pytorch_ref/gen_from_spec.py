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
- `bf16p`: both inputs and outputs stored as native `torch.bfloat16` (2 bytes
  per element). Matches `WgslConfig::BF16_PACKED`: `array<u32>` activation
  storage with 2 elems/word. Inputs in this dtype are not the same bytes as
  the fp32 fixture's inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def quantize_for_dtype(t: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Encode a fp32 tensor for the wire format the GPU buffer expects in
    this dtype. fp32 stays fp32 bytes; bf16w stays fp32 layout with rounded
    values; bf16p emits native 2-byte bf16."""
    if dtype_name == "fp32":
        return t
    if dtype_name == "bf16w":
        return t.to(torch.bfloat16).to(torch.float32)
    if dtype_name == "bf16p":
        return t.to(torch.bfloat16)
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
    if op == "relu":
        return F.relu(inputs["x"])
    if op == "memcat":
        # MemBlock input assembly: x is [T, C, H, W]; concat current frame with
        # the previous frame (zero at t=0) on the channel axis -> [T, 2C, H, W].
        x = inputs["x"]
        past = torch.zeros_like(x)
        past[1:] = x[:-1]
        return torch.cat([x, past], dim=1)
    if op == "silu_mul":
        return F.silu(inputs["a"]) * inputs["b"]
    if op == "gelu_mul":
        # gelu_new (tanh approximation) gate, matching HF NewGELUActivation.
        return F.gelu(inputs["a"], approximate="tanh") * inputs["b"]
    if op == "tanh":
        return torch.tanh(inputs["x"])
    if op == "bcast_affine":
        return inputs["x"] * (inputs["s"] + float(case["bias"]))
    if op == "bcast_fma":
        return inputs["x"] + inputs["s"] * inputs["y"]
    if op == "bcast_modulate":
        return inputs["x"] * (inputs["s"] + float(case["bias"])) + inputs["t"]
    if op == "bcast_add":
        return inputs["x"] + inputs["s"]
    if op == "bcast_mul":
        return inputs["x"] * inputs["s"]
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
    if op == "conv3d":
        # Causal time conv: pad only the front of the time axis (low-time
        # side) by pad_t; H/W stay symmetric. F.pad order for NCTHW is
        # (w_l, w_r, h_l, h_r, t_l, t_r).
        x = F.pad(inputs["x"], (0, 0, 0, 0, int(case["pad_t"]), 0))
        return F.conv3d(
            x,
            inputs["w"],
            bias=inputs["bias"],
            stride=(int(case["stride_t"]), int(case["stride_h"]), int(case["stride_w"])),
            padding=(0, int(case["pad_h"]), int(case["pad_w"])),
        )
    if op == "conv1d":
        return F.conv1d(
            inputs["x"],
            inputs["w"],
            bias=inputs["bias"],
            stride=int(case["stride"]),
            padding=int(case["pad"]),
            dilation=int(case["dilation"]),
            groups=int(case["groups"]),
        )
    if op == "conv_transpose1d":
        return F.conv_transpose1d(
            inputs["x"],
            inputs["w"],
            bias=inputs["bias"],
            stride=int(case["stride"]),
            padding=int(case["pad"]),
            dilation=int(case["dilation"]),
            groups=int(case["groups"]),
        )
    if op == "snake_beta":
        # BigVGAN v2 SnakeBeta (log-scale alpha/beta): per-channel over NCL.
        x = inputs["x"]
        eps = float(case["eps"])
        c = x.shape[1]
        alpha = torch.exp(inputs["alpha"]).reshape(1, c, 1)
        beta = torch.exp(inputs["beta"]).reshape(1, c, 1)
        return x + (1.0 / (beta + eps)) * torch.sin(x * alpha).pow(2)
    if op == "replicate_pad1d":
        return F.pad(inputs["x"], (int(case["lpad"]), int(case["rpad"])), mode="replicate")
    if op == "scale":
        return inputs["x"] * float(case["scale"])
    if op == "rmsnorm3d":
        # WanRMS_norm: L2-normalize across the channel axis (dim 1 of NCTHW),
        # scale by sqrt(C), apply per-channel gain. bias=False in the Wan VAE.
        x = inputs["x"]
        gamma = inputs["w"]
        c = x.shape[1]
        normed = F.normalize(x, dim=1)
        return normed * (c**0.5) * gamma.reshape(1, c, 1, 1, 1)
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
    if op == "gated_head_mul":
        # out[i] = x[i] * 2*sigmoid(gate[i // head_dim]); head_dim = len(x)/len(gate).
        x = inputs["x"].reshape(-1)
        gate = inputs["gate"].reshape(-1)
        head_dim = x.numel() // gate.numel()
        g = gate.repeat_interleave(head_dim)
        return x * (2.0 * torch.sigmoid(g))
    if op == "pixel_norm3d":
        # Weightless per-location channel-RMS over NCTHW (eps inside the mean).
        x = inputs["x"]
        eps = float(case["eps"])
        ms = x.pow(2).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(ms + eps)
    if op == "depth_to_space3d":
        # einops b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3); drop the
        # leading frame when p1 == 2. Input here is [cin, t, h, w] (B=1).
        x = inputs["x"]
        p1, p2, p3 = int(case["p1"]), int(case["p2"]), int(case["p3"])
        cin, t, h, w = x.shape
        cout = cin // (p1 * p2 * p3)
        x = x.reshape(cout, p1, p2, p3, t, h, w)
        x = x.permute(0, 4, 1, 5, 2, 6, 3)  # cout, t, p1, h, p2, w, p3
        x = x.reshape(cout, t * p1, h * p2, w * p3)
        if p1 == 2:
            x = x[:, 1:, :, :]  # drop leading frame
        return x.contiguous()
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
        inputs: dict[str, torch.Tensor] = {}
        for inp in case["inputs"]:
            inputs[inp["name"]] = make_input(inp["fill"], inp["shape"])

        for dtype_name in case["dtypes"]:
            bucket = by_dtype.setdefault(dtype_name, {})
            # For bf16p, the shader reads bf16-packed inputs and unpacks to
            # fp32 before compute. Ref pytorch mirrors that: bf16-round inputs
            # first, recompute in fp32, then bf16-encode output. bf16w keeps
            # inputs fp32 (storage is fp32 in that path; only writes round).
            if dtype_name == "bf16p":
                rounded = {k: v.to(torch.bfloat16).to(torch.float32) for k, v in inputs.items()}
                out_fp32 = compute_output(case["op"], case, rounded)
                for k, v in inputs.items():
                    bucket[f"{name}/{k}"] = v.to(torch.bfloat16).contiguous()
                bucket[f"{name}/out"] = out_fp32.to(torch.bfloat16).contiguous()
            else:
                out_fp32 = compute_output(case["op"], case, inputs)
                for k, v in inputs.items():
                    bucket[f"{name}/{k}"] = v.contiguous()
                bucket[f"{name}/out"] = quantize_for_dtype(out_fp32, dtype_name).contiguous()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dtype_name, tensors in by_dtype.items():
        save_file(tensors, str(args.out_dir / f"{dtype_name}.safetensors"))


if __name__ == "__main__":
    main()
