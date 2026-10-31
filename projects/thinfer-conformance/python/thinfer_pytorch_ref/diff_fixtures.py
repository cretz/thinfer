"""Compare two fixture directories (committed vs freshly generated).

Byte equality only holds on one machine: torch's CPU kernels differ in the
last ulps across CPUs and OSes. So this checks what makes a fixture stale -
the same files, tensor names, shapes and dtypes - and allows values only
ulp-level noise: per element, `|a - b| <= rtol * (|b| + max|b|)`, with a
few ulps of the file's stored precision. A generator or spec change moves
values by far more than that.

Usage: python -m thinfer_pytorch_ref.diff_fixtures <committed> <fresh>
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Rounding noise in each file's stored precision (fp32 ulp: 2^-23; bf16: 2^-8).
# fp32 gets more room: a long reduction (conv3d multitile) sums in a different
# order on another CPU and lands ~40 ulps away (seen: 4.6e-6 relative on CI).
RTOL = {"fp32": 2.0**-16, "bf16w": 2.0**-6, "bf16p": 2.0**-6}


def main() -> None:
    committed, fresh = Path(sys.argv[1]), Path(sys.argv[2])
    names = sorted(p.name for p in committed.glob("*.safetensors"))
    fresh_names = sorted(p.name for p in fresh.glob("*.safetensors"))
    failures: list[str] = []
    if names != fresh_names:
        failures.append(f"files differ: committed {names}, fresh {fresh_names}")
    for name in sorted(set(names) & set(fresh_names)):
        rtol = RTOL[name.removesuffix(".safetensors")]
        a, b = load_file(committed / name), load_file(fresh / name)
        if a.keys() != b.keys():
            failures.append(f"{name}: tensors differ: {sorted(a.keys() ^ b.keys())}")
        worst, worst_key, noisy = 0.0, "", 0
        for key in sorted(a.keys() & b.keys()):
            ta, tb = a[key], b[key]
            if ta.shape != tb.shape or ta.dtype != tb.dtype:
                failures.append(
                    f"{name}:{key}: {ta.dtype}{list(ta.shape)} vs {tb.dtype}{list(tb.shape)}"
                )
                continue
            if torch.equal(ta, tb):
                continue
            noisy += 1
            ta, tb = ta.double(), tb.double()
            diff = (ta - tb).abs()
            bound = rtol * (tb.abs() + tb.abs().max())
            # Equal elements are 0 even where the bound is (0/0 would be NaN,
            # and a NaN max would hide every other element).
            ratio = torch.where(diff == 0, 0.0, diff / bound).max().item()
            if ratio > worst:
                worst, worst_key = ratio, key
            if ratio > 1.0:
                failures.append(f"{name}:{key}: {ratio:.2f}x the noise bound")
        print(f"{name}: {len(a)} tensors, {noisy} not bit-equal, worst {worst:.3f}x bound ({worst_key})")
    for f in failures:
        print(f"STALE {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
