"""Tokenize a prompt with the PRODUCT Gemma-3 tokenizer.json (no model load).

The e2e health driver runs the engine encoder itself; it only needs the token
ids the engine feeds. The engine wraps the HF `tokenizers` crate over the product
`tokenizer.json` (`HfTokenizer::encode(prompt.trim(), true)`), so mirror it here:
`tokenizers.Tokenizer.from_file(...)`, `prompt.strip()`, `add_special_tokens=True`.
NOT `AutoTokenizer(gguf_file=...)`, whose GGUF-reconstructed tokenizer is degenerate
(splits words into char-fragments, ~2x the token count) and is NOT what the engine
or serve runs. Cheap: loads only the tokenizer, not Gemma.

Dumps:
  token_ids.bin   u32 [n]
  tokens.txt      "n"

  uv run --with tokenizers python -m \\
      thinfer_pytorch_ref.ltx.gen_tokens_ref --tokenizer <tokenizer.json> \\
      --prompt "<text>" --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, type=Path, help="product tokenizer.json")
    p.add_argument("--prompt", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = tok.encode(args.prompt.strip(), add_special_tokens=True).ids
    print(f"[ltx-tokens] prompt -> {len(ids)} tokens: {ids[:12]}...", flush=True)
    np.asarray(ids, dtype="<u4").tofile(str(args.out / "token_ids.bin"))
    (args.out / "tokens.txt").write_text(f"{len(ids)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
