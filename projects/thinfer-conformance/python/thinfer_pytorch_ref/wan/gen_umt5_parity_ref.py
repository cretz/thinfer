"""umT5-XXL text-encoder parity reference: token ids -> per-layer encoder
hidden states plus per-op intermediates for selected layers. Pairs with the
engine umT5 encoder; the Rust side feeds the SAME token ids and linfits every
dump below against the matching engine tap.

umT5 is T5-family (encoder-decoder; we only run the encoder), so the parity
shape differs from Qwen3 in ways that matter for bit-clean math:

  - RMSNorm without mean subtraction and without bias (T5LayerNorm).
  - Self-attention has NO 1/sqrt(d) query scaling (T5 folds scale into init).
  - Per-layer relative-position bias (umT5 / "scalable_attention"), unlike
    vanilla T5 which shares layer 0's bias. Added to the attention logits.
  - Gated-GELU FF: wi_0 (gate, gelu_new) * wi_1 (up) -> wo (down).

Dumps (little-endian f32, full tensors; seq is the 512 text context):

  py_umt5_hs{i}.bin      i in 0..N_LAYERS. hs[0] = embeddings, hs[k] =
                         encoder-block-(k-1) output. hs[N_LAYERS] is the
                         post-final_layer_norm output (the encoder result the
                         DiT cross-attends to); dumped because the engine DOES
                         run the final norm here (unlike Qwen3).
  py_umt5_l{n}_{op}.bin  per-op taps for each layer n in --tap-layers:
                         n1 (self-attn layer_norm), q, k, v, sa (o input),
                         proj (o output), bias (relative position bias),
                         n2 (FF layer_norm), gate (wi_0), up (wi_1), down (wo).

Usage:

    uv run python -m thinfer_pytorch_ref.wan.gen_umt5_parity_ref \\
        --token-ids <ids.bin (u32 LE)> --out <dir> [--tap-layers 0,12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

# Diffusers SkyReels-V2 bundle carries the umT5-XXL encoder under
# `text_encoder`; loading from here (not google/umt5-xxl) keeps the reference
# on the exact checkpoint the pipeline ships.
REPO = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--token-ids", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tap-layers", default="0")
    p.add_argument("--dtype", choices=list(DTYPES.keys()), default="bf16")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tap_layers = sorted({int(s) for s in args.tap_layers.split(",") if s})

    from transformers import UMT5EncoderModel

    ids = np.fromfile(args.token_ids, dtype="<u4").astype(np.int64)
    input_ids = torch.from_numpy(ids)[None, :]
    print(f"[py-umt5] tokens={ids.size} tap_layers={tap_layers}", flush=True)

    # bf16 reference at ~12 GB RAM, matching the Z-Image pyref footprint (the
    # q8 engine path is diffed against a bf16 reference, so fp32 here buys no
    # parity, only RAM). low_cpu_mem_usage mmaps the safetensors and
    # materializes straight to the target dtype, avoiding an fp32 load spike.
    model = UMT5EncoderModel.from_pretrained(
        REPO,
        subfolder="text_encoder",
        torch_dtype=DTYPES[args.dtype],
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[py-umt5] loaded encoder dtype={args.dtype}", flush=True)

    cap: dict[str, torch.Tensor] = {}

    def grab(name: str):
        def hook(_m, _i, out):
            cap[name] = out if isinstance(out, torch.Tensor) else out[0]

        return hook

    for n in tap_layers:
        blk = model.encoder.block[n]
        attn = blk.layer[0].SelfAttention
        blk.layer[0].layer_norm.register_forward_hook(grab(f"l{n}_n1"))
        attn.q.register_forward_hook(grab(f"l{n}_q"))
        attn.k.register_forward_hook(grab(f"l{n}_k"))
        attn.v.register_forward_hook(grab(f"l{n}_v"))
        attn.o.register_forward_pre_hook(
            lambda _m, inp, _n=n: cap.__setitem__(f"l{_n}_sa", inp[0])
        )
        attn.o.register_forward_hook(grab(f"l{n}_proj"))
        ff = blk.layer[1]
        ff.layer_norm.register_forward_hook(grab(f"l{n}_n2"))
        ff.DenseReluDense.wi_0.register_forward_hook(grab(f"l{n}_gate"))
        ff.DenseReluDense.wi_1.register_forward_hook(grab(f"l{n}_up"))
        ff.DenseReluDense.wo.register_forward_hook(grab(f"l{n}_down"))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    # output_hidden_states gives len = num_layers + 1: hs[0] = embeddings,
    # hs[1..L-1] = per-block outputs, hs[L] = post-final_layer_norm (==
    # last_hidden_state, the encoder result the DiT cross-attends to).
    hs = out.hidden_states
    n_layers = model.config.num_layers
    assert len(hs) == n_layers + 1, f"hidden_states len {len(hs)}"
    for i in range(len(hs)):
        _dump(hs[i], args.out / f"py_umt5_hs{i}.bin")

    for name, t in cap.items():
        _dump(t, args.out / f"py_umt5_{name}.bin")

    # Relative-position bias per tapped layer. umT5 recomputes it inside each
    # SelfAttention.compute_bias; recompute manually so the engine can compare
    # its precomputed bias table. Version-drift tolerant: a failure here only
    # loses the bias tap (q/k/v/sa still bracket the segment).
    try:
        seq = ids.size
        for n in tap_layers:
            attn = model.encoder.block[n].layer[0].SelfAttention
            bias = attn.compute_bias(seq, seq, device=input_ids.device)
            _dump(bias.contiguous(), args.out / f"py_umt5_l{n}_bias.bin")
    except Exception as e:  # noqa: BLE001
        print(f"[py-umt5] bias recompute skipped: {e}", flush=True)

    print("[py-umt5] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
