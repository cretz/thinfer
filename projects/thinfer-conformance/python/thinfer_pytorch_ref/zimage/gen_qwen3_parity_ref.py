"""Qwen3 text-encoder parity reference: token ids -> per-layer hidden states
plus per-op intermediates for selected layers. Pairs with
`tests/zimage/qwen3_parity.rs`, which feeds the SAME (even-padded) token ids
to the engine encoder and linfits every dump below against the matching
engine tap.

Dumps (little-endian f32, full tensors; seq is tiny):

  py_qwen3_hs{i}.bin      i in 0..N_LAYERS-1. hs[0] = embeddings,
                          hs[k] = layer-(k-1) output. The post-final-norm
                          hidden_states[-1] is NOT dumped (the engine never
                          runs `model.norm`).
  py_qwen3_l{n}_{op}.bin  per-op taps for each layer n in --tap-layers:
                          n1, q, k, v, qn, kn, qr, kr, sa (o_proj input),
                          proj, n2, gate, up, down. qr/kr are recomputed
                          manually via `apply_rotary_pos_emb` (rope has no
                          module to hook).

Usage:

    uv run python -m thinfer_pytorch_ref.zimage.gen_qwen3_parity_ref \\
        --token-ids <ids.bin (u32 LE)> --out <dir> [--tap-layers 0,6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = "Tongyi-MAI/Z-Image-Turbo"


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--token-ids", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tap-layers", default="0")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tap_layers = sorted({int(s) for s in args.tap_layers.split(",") if s})

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    ids = np.fromfile(args.token_ids, dtype="<u4").astype(np.int64)
    input_ids = torch.from_numpy(ids)[None, :]
    print(f"[py-qwen3] tokens={ids.size} tap_layers={tap_layers}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    model.eval()

    cap: dict[str, torch.Tensor] = {}

    def grab(name: str):
        def hook(_m, _i, out):
            cap[name] = out if isinstance(out, torch.Tensor) else out[0]

        return hook

    for n in tap_layers:
        ln = model.model.layers[n]
        ln.input_layernorm.register_forward_hook(grab(f"l{n}_n1"))
        ln.self_attn.q_proj.register_forward_hook(grab(f"l{n}_q"))
        ln.self_attn.k_proj.register_forward_hook(grab(f"l{n}_k"))
        ln.self_attn.v_proj.register_forward_hook(grab(f"l{n}_v"))
        ln.self_attn.q_norm.register_forward_hook(grab(f"l{n}_qn"))
        ln.self_attn.k_norm.register_forward_hook(grab(f"l{n}_kn"))
        ln.self_attn.o_proj.register_forward_pre_hook(
            lambda _m, inp, _n=n: cap.__setitem__(f"l{_n}_sa", inp[0])
        )
        ln.self_attn.o_proj.register_forward_hook(grab(f"l{n}_proj"))
        ln.post_attention_layernorm.register_forward_hook(grab(f"l{n}_n2"))
        ln.mlp.gate_proj.register_forward_hook(grab(f"l{n}_gate"))
        ln.mlp.up_proj.register_forward_hook(grab(f"l{n}_up"))
        ln.mlp.down_proj.register_forward_hook(grab(f"l{n}_down"))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    n_layers = model.config.num_hidden_layers
    assert len(hs) == n_layers + 1, f"hidden_states len {len(hs)}"
    for i in range(n_layers):
        _dump(hs[i], args.out / f"py_qwen3_hs{i}.bin")

    for name, t in cap.items():
        _dump(t, args.out / f"py_qwen3_{name}.bin")

    # Manual rope recompute -> qr/kr (engine taps the post-rope q/k). The
    # q_norm/k_norm hook outputs are [b, s, h, hd]; apply_rotary_pos_emb
    # wants [b, h, s, hd]; engine layout is s-major, so transpose back
    # before dumping. Version-drift tolerant: a failure here only loses the
    # rope/sdpa split (the sa tap still brackets the segment).
    try:
        pos = torch.arange(ids.size, dtype=torch.long)[None, :]
        cos, sin = model.model.rotary_emb(hs[0], pos)
        for n in tap_layers:
            q_r, k_r = apply_rotary_pos_emb(
                cap[f"l{n}_qn"].transpose(1, 2),
                cap[f"l{n}_kn"].transpose(1, 2),
                cos,
                sin,
            )
            _dump(q_r.transpose(1, 2).contiguous(), args.out / f"py_qwen3_l{n}_qr.bin")
            _dump(k_r.transpose(1, 2).contiguous(), args.out / f"py_qwen3_l{n}_kr.bin")
    except Exception as e:  # noqa: BLE001
        print(f"[py-qwen3] rope recompute skipped: {e}", flush=True)

    print("[py-qwen3] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
