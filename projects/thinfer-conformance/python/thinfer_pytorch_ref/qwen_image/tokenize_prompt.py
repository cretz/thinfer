"""Tokenize a prompt with the Qwen-Image t2i template (tokenizer only, no model
load). Dumps `token_ids.bin` (u32 LE) so the Rust e2e feeds the engine encoder
the exact ids the diffusers pipeline would.

Usage:

    uv run python -m thinfer_pytorch_ref.qwen_image.tokenize_prompt \\
        --tokenizer-dir <dir> --prompt "..." --out <token_ids.bin>
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    text = PROMPT_TEMPLATE.format(args.prompt)
    ids = tok(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ids.astype("<u4").tofile(str(args.out))
    print(f"[py-qwen-tok] {len(ids)} tokens -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
