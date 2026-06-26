"""Tokenize a prompt with the Gemma-3 GGUF tokenizer (no model load).

The e2e health driver runs the engine encoder itself; it only needs the token
ids the upstream tokenizer produces (`AutoTokenizer.from_pretrained(dir,
gguf_file=...)`, `prompt.strip()`, `add_special_tokens=True`) -- the SAME path
`gen_encoder_ref.py` uses. Cheap: loads only the tokenizer, not Gemma.

Dumps:
  token_ids.bin   u32 [n]
  tokens.txt      "n"

  uv run --with gguf --with transformers python -m \\
      thinfer_pytorch_ref.ltx.gen_tokens_ref --gguf <gemma...gguf> \\
      --prompt "<text>" --out <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--prompt", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.dirname(str(args.gguf)), gguf_file=os.path.basename(str(args.gguf))
    )
    ids = tok(args.prompt.strip(), add_special_tokens=True)["input_ids"]
    print(f"[ltx-tokens] prompt -> {len(ids)} tokens: {ids[:12]}...", flush=True)
    np.asarray(ids, dtype="<u4").tofile(str(args.out / "token_ids.bin"))
    (args.out / "tokens.txt").write_text(f"{len(ids)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
