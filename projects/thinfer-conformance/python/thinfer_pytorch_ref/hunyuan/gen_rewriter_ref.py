"""HunyuanVideo prompt-rewriter (Qwen3-VL-8B-Instruct) greedy-decode oracle.

Loads the text-only rewriter GGUF via `llama-cpp-python` (CPU), builds the exact
ChatML prompt, tokenizes it, and greedy-decodes N tokens. Pairs with
`tests/qwen3_lm/logits_parity.rs`, which teacher-forces the engine's full causal
forward against `greedy_ids` and checks per-step argmax agreement.

Dumps (little-endian u32, into `--out`):

  prompt_ids.bin   the ChatML prompt token ids (system + user + assistant tag).
  greedy_ids.bin   the first N greedy (temp=0, top_k=1) continuation token ids,
                   truncated at EOS.

Run (CPU, no GPU):

    uv run --with llama-cpp-python python -m \\
        thinfer_pytorch_ref.hunyuan.gen_rewriter_ref \\
        --gguf <Q5_K_M.gguf> --system-file <sys.txt> \\
        --user "a cat sleeps" --out <dir> --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _dump_u32(ids: list[int], path: Path) -> None:
    np.asarray(ids, dtype="<u4").tofile(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--system-file", required=True, type=Path)
    p.add_argument("--user", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", default=24, type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from llama_cpp import Llama

    sys = args.system_file.read_text(encoding="utf-8")

    llm = Llama(
        model_path=str(args.gguf),
        n_ctx=8192,
        n_gpu_layers=0,
        logits_all=False,
        verbose=False,
    )

    # ChatML, byte-for-byte the layout the engine test tokenizes.
    prompt = (
        f"<|im_start|>system\n{sys}<|im_end|>\n"
        f"<|im_start|>user\n{args.user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Count the system-only tokens for the anchor sanity print.
    sys_ids = llm.tokenize(sys.encode("utf-8"), add_bos=False, special=True)
    prompt_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)

    eos = llm.token_eos()
    greedy: list[int] = []
    for tok in llm.generate(prompt_ids, temp=0.0, top_k=1):
        if tok == eos or len(greedy) >= args.n:
            break
        greedy.append(int(tok))

    _dump_u32([int(t) for t in prompt_ids], args.out / "prompt_ids.bin")
    _dump_u32(greedy, args.out / "greedy_ids.bin")

    text = llm.detokenize(greedy).decode("utf-8", errors="replace")
    print(
        f"[rewriter-ref] system_tokens={len(sys_ids)} prompt_tokens={len(prompt_ids)} "
        f"n_greedy={len(greedy)} first_greedy_ids={greedy[:8]} "
        f"decoded={text!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
