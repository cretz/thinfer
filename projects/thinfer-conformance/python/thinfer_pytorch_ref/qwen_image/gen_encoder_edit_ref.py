"""Qwen-Image-Edit text-encoder parity reference (LM + 3-axis MRoPE + scatter).

Isolates the LM edit path from the vision tower: this pyref runs the vision
tower itself (to produce + dump the merged vision embeds), then runs the
Qwen2.5-VL-7B LM on the edit-templated prompt with those embeds scattered into
the `<|image_pad|>` slots and the 3D MRoPE position_ids the classic
get_rope_index scheme produces. The ENGINE consumes the SAME dumped vision
embeds, so the comparison isolates the LM + MRoPE + scatter kernels.

Both towers load by DEQUANTIZING the SAME GGUFs the engine loads (Q8_0 LM ->
bf16, f16 mmproj), so parity isolates kernel correctness from quant loss.

  token_ids.bin     u32 LE, the edit-templated prompt tokens (image_pad expanded).
  vision_embeds.bin f32 LE [n_img, 3584], the merged vision-tower output (= what
                    HF scatters into the image-pad block).
  hidden.bin        f32 LE [seq, 3584], LM hidden_states[-1] (post output_norm).
  meta.txt          "seq hidden image_pad_start n_img mgh mgw"

LM is 7B bf16 (~15GB); run as its own process so it frees on exit.

Usage:

    uv run --with gguf python -m thinfer_pytorch_ref.qwen_image.gen_encoder_edit_ref \\
        --gguf <...Qwen2.5-VL-7B...Q8_0.gguf> --mmproj <...mmproj-f16.gguf> \\
        --tokenizer-dir <dir> --prompt "..." --out <dir> --gh 8 --gw 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from thinfer_pytorch_ref.qwen_image.gen_encoder_ref import (
    EPS,
    FFN,
    HEAD_DIM,
    HIDDEN,
    MROPE_SECTION,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    ROPE_THETA,
    VOCAB,
    _load_gguf_into,
)
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import (
    MERGE,
    OUT_HIDDEN,
    _build_state_dict,
    _gguf_tensors,
    _patchify,
)
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import DEPTH as V_DEPTH
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import HEADS as V_HEADS
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import HIDDEN as V_HIDDEN
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import INTERMEDIATE as V_INTERMEDIATE
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import PATCH as V_PATCH
from thinfer_pytorch_ref.qwen_image.gen_vision_ref import TEMPORAL as V_TEMPORAL

IMAGE_PAD = "<|image_pad|>"
DROP_IDX = 64  # prompt_template_encode_start_idx (edit template system preamble)

# Edit template (pipeline_qwenimage_edit.py); the single <|image_pad|> expands to
# n_img = mgh*mgw placeholders before tokenization.
EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, "
    "shape, size, texture, objects, background), then explain how the user's "
    "text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n<|im_start|>user\n"
    "<|vision_start|>{img}<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
)


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _run_vision(mmproj: Path, gh: int, gw: int) -> np.ndarray:
    """Build the vision tower from the mmproj GGUF, run the deterministic
    patchified image, return merged embeds [n_img, 3584] (raster order)."""
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    cfg = Qwen2_5_VLVisionConfig(
        depth=V_DEPTH,
        hidden_size=V_HIDDEN,
        num_heads=V_HEADS,
        intermediate_size=V_INTERMEDIATE,
        out_hidden_size=OUT_HIDDEN,
        spatial_merge_size=MERGE,
        patch_size=V_PATCH,
        temporal_patch_size=V_TEMPORAL,
        in_channels=3,
        window_size=112,
        fullatt_block_indexes=[7, 15, 23, 31],
        hidden_act="silu",
    )
    model = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval().to(torch.float32)
    model.config._attn_implementation = "eager"
    sd = _build_state_dict(_gguf_tensors(mmproj))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    assert not missing, f"vision missing: {sorted(missing)[:8]}"
    assert not unexpected, f"vision unexpected: {sorted(unexpected)[:8]}"

    pixel_values = _patchify(gh, gw)
    grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long)
    out = model(hidden_states=torch.from_numpy(pixel_values).float(), grid_thw=grid_thw)
    embeds = out.pooler_output  # [n_img, 3584]
    n_img = (gh * gw) // (MERGE * MERGE)
    assert embeds.shape == (n_img, OUT_HIDDEN), embeds.shape
    return embeds.to(torch.float32).cpu().numpy().astype("float32")


def _edit_position_ids(seq: int, pad_start: int, mgh: int, mgw: int) -> torch.Tensor:
    """The classic Qwen2.5-VL get_rope_index scheme for one leading image:
    leading text 0..pad_start on all axes; image t const, h=cur+row, w=cur+col
    over the merged grid (row-major h-then-w); cur += max(mgh, mgw); trailing
    text sequential. Returns [3, 1, seq]."""
    n_img = mgh * mgw
    pos = torch.zeros(3, seq, dtype=torch.long)
    for i in range(pad_start):
        pos[:, i] = i
    cur = pad_start
    for r in range(mgh):
        for c in range(mgw):
            i = pad_start + r * mgw + c
            pos[0, i] = cur
            pos[1, i] = cur + r
            pos[2, i] = cur + c
    nxt = cur + max(mgh, mgw)
    for i in range(pad_start + n_img, seq):
        pos[:, i] = nxt
        nxt += 1
    return pos.view(3, 1, seq)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--mmproj", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gh", type=int, default=8)
    p.add_argument("--gw", type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.gh % MERGE == 0 and args.gw % MERGE == 0, "grid must be /2"

    from transformers import AutoTokenizer
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLTextModel,
    )

    mgh, mgw = args.gh // MERGE, args.gw // MERGE
    n_img = mgh * mgw

    # --- vision tower first (fp32, ~0.6B), dump merged embeds ---
    torch.set_grad_enabled(False)
    print(f"[py-qwen-edit] vision tower {args.gh}x{args.gw} -> {n_img} tokens", flush=True)
    vision_embeds = _run_vision(args.mmproj, args.gh, args.gw)
    _dump(torch.from_numpy(vision_embeds), args.out / "vision_embeds.bin")

    # --- tokenize the edit prompt (image_pad expanded to n_img) ---
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    text = EDIT_TEMPLATE.format(img=IMAGE_PAD * n_img, prompt=args.prompt)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]  # [1, L]
    seq = int(input_ids.shape[1])
    ids_row = input_ids[0].tolist()
    image_token_id = 151655
    pad_positions = [i for i, t in enumerate(ids_row) if t == image_token_id]
    assert len(pad_positions) == n_img, f"{len(pad_positions)} image-pad != n_img {n_img}"
    pad_start = pad_positions[0]
    assert pad_positions == list(range(pad_start, pad_start + n_img)), "image-pad not contiguous"
    input_ids.to(torch.int32).numpy().astype("<u4").tofile(str(args.out / "token_ids.bin"))
    print(f"[py-qwen-edit] prompt -> {seq} tokens, image_pad_start={pad_start}", flush=True)

    # --- LM (7B), load the SAME GGUF (dequant Q8_0 -> bf16) ---
    cfg = Qwen2_5_VLTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=FFN,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        tie_word_embeddings=False,
        rope_scaling={"rope_type": "default", "mrope_section": MROPE_SECTION, "rope_theta": ROPE_THETA},
    )
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen2_5_VLTextModel(cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    print("[py-qwen-edit] loading LM GGUF (dequant Q8_0 -> bf16)...", flush=True)
    _load_gguf_into(model, args.gguf)

    # --- inputs_embeds = embed lookup, then scatter vision embeds ---
    embed_tokens = model.get_input_embeddings()
    inputs_embeds = embed_tokens(input_ids)  # [1, seq, HIDDEN] bf16
    vis = torch.from_numpy(vision_embeds).to(inputs_embeds.dtype)
    inputs_embeds[0, pad_start : pad_start + n_img, :] = vis

    # --- 3D MRoPE positions (classic single-image scheme) ---
    pos = _edit_position_ids(seq, pad_start, mgh, mgw)  # [3, 1, seq]

    out = model(
        inputs_embeds=inputs_embeds,
        position_ids=pos,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = out.hidden_states[-1][0]  # [seq, HIDDEN] post output_norm
    _dump(hidden, args.out / "hidden.bin")
    (args.out / "meta.txt").write_text(f"{seq} {HIDDEN} {pad_start} {n_img} {mgh} {mgw}\n")
    print(
        f"[py-qwen-edit] done: hidden{tuple(hidden.shape)} "
        f"range[{hidden.min():.3f},{hidden.max():.3f}] drop_idx={DROP_IDX}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
