"""Ideogram-4 end-to-end parity reference, STAGED across processes.

The full pipeline (encode -> DiT denoise loop -> VAE decode) would co-reside
the 8B encoder (~16GB bf16) and the 9B DiT (~18GB bf16); to respect host RAM
the Rust gate invokes this module THREE times, once per `--stage`, so each
model frees on process exit (peak ~ one model). Intermediates flow through the
`--out` dir:

  stage encode:  prompt -> token_ids.bin (u32), llm_features.bin (f32
                 [num_text, 53248]), meta_encode.txt ("num_text").
  stage dit:     llm_features + the LoRA-folded DiT -> noise.bin (f32
                 [num_image, 128], the seeded `torch.randn` the engine injects),
                 latent.bin (f32 [num_image, 128], the denoised latent),
                 meta_dit.txt ("num_image grid_h grid_w").
  stage vae:     latent -> decoded.bin (f32 [3, H, W] in [-1, 1], pre-uint8).

The DiT stage dequantizes the SAME DiT GGUF the engine loads and folds the SAME
turbotime LoRA (`W += B@A`, scale 1.0) so the only band is the engine's f16-act
vs this bf16/fp32 staged reference. No-CFG (the LoRA drops the unconditional
branch): `v = pos_v`, `z += v * (s_val - t_val)`.

Usage (per stage):

    uv run --with gguf python -m thinfer_pytorch_ref.ideogram4.gen_e2e_ref \\
        --stage encode --enc-gguf <q8.gguf> --out <dir> --prompt "..."
    uv run --with gguf python -m thinfer_pytorch_ref.ideogram4.gen_e2e_ref \\
        --stage dit --dit-gguf <q8.gguf> --lora <turbotime.safetensors> \\
        --out <dir> --width 64 --height 64 --steps 4 --mu 0.5 --std 1.75 --seed 0
    uv run --with einops python -m thinfer_pytorch_ref.ideogram4.gen_e2e_ref \\
        --stage vae --vae <FLUX.2-VAE.safetensors> --out <dir> \\
        --width 64 --height 64
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

PATCH = 16  # patch_size(2) * ae_scale_factor(8)
PATCH_SIZE = 2
AE_CHANNELS = 32
IMAGE_POSITION_OFFSET = 65536
LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2
REPO = "Qwen/Qwen3-VL-8B-Instruct"
TAP_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
# The six per-layer matmul sites the turbotime LoRA touches.
LORA_SITES = [
    "adaln_modulation",
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
]


def _find_third_party() -> Path:
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "third-party" / "ideogram4" / "src"
        if cand.exists():
            return cand
    raise FileNotFoundError("third-party/ideogram4/src not found above this file")


def _stub_ideogram4_pkg() -> Path:
    """Register a lightweight `ideogram4` package (submodule imports only)."""
    src = _find_third_party()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "ideogram4" not in sys.modules:
        pkg = types.ModuleType("ideogram4")
        pkg.__path__ = [str(src / "ideogram4")]
        sys.modules["ideogram4"] = pkg
    return src


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _dequantize(reader_tensor) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    if deq is not None:
        arr = deq(reader_tensor.data, reader_tensor.tensor_type)
    else:
        arr = gguf.quants.dequantize(reader_tensor.data, reader_tensor.tensor_type)
    return np.ascontiguousarray(arr).astype(np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# stage: encode
# --------------------------------------------------------------------------- #
def _gguf_to_hf_key(name: str) -> str | None:
    sites = {
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_q_norm": "self_attn.q_norm",
        "attn_k_norm": "self_attn.k_norm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    if name == "token_embd.weight":
        return "embed_tokens.weight"
    if name == "output_norm.weight":
        return "norm.weight"
    if name == "output.weight":
        return None
    if name.startswith("blk."):
        _, idx, rest = name.split(".", 2)
        site = rest.rsplit(".", 1)[0]
        hf = sites.get(site)
        if hf is None:
            raise ValueError(f"unmapped GGUF site {site!r} in {name!r}")
        return f"layers.{idx}.{hf}.weight"
    raise ValueError(f"unexpected GGUF tensor {name!r}")


def _load_encoder_gguf(model: torch.nn.Module, gguf_path: Path) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()
    for t in reader.tensors:
        key = _gguf_to_hf_key(t.name)
        if key is None:
            continue
        param = state.get(key)
        if param is None:
            raise KeyError(f"GGUF tensor {t.name!r} -> {key!r} not in model")
        flat = _dequantize(t)
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(key)
        del flat
    missing = {k for k in (set(state) - seen) if "rotary" not in k}
    if missing:
        raise RuntimeError(f"encoder GGUF missing {len(missing)}: {sorted(missing)[:5]}")


def stage_encode(args) -> int:
    from transformers import AutoTokenizer, Qwen3VLConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLTextRotaryEmbedding,
    )

    tokenizer = AutoTokenizer.from_pretrained(REPO)
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    seq = int(input_ids.shape[1])
    input_ids.to(torch.int32).numpy().astype("<u4").tofile(str(args.out / "token_ids.bin"))
    print(f"[e2e-encode] prompt -> {seq} tokens", flush=True)

    cfg = Qwen3VLConfig.from_pretrained(REPO)
    text_cfg = cfg.get_text_config()
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3VLTextModel(text_cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(text_cfg)
    print("[e2e-encode] loading encoder GGUF (dequant Q8 -> bf16)...", flush=True)
    _load_encoder_gguf(model, args.enc_gguf)

    inputs_embeds = model.embed_tokens(input_ids)
    pos = torch.arange(seq).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids = pos[0]
    mrope_position_ids = pos[1:]
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(TAP_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    taps = [captured[i][0] for i in TAP_LAYERS]  # each [seq, H]
    stacked = torch.stack(taps, dim=0).permute(1, 2, 0).reshape(seq, -1)  # [seq, H*13]
    _dump(stacked, args.out / "llm_features.bin")
    (args.out / "meta_encode.txt").write_text(f"{seq}\n")
    print(f"[e2e-encode] done: feats {tuple(stacked.shape)}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# stage: dit
# --------------------------------------------------------------------------- #
def _import_modeling():
    _stub_ideogram4_pkg()
    from ideogram4 import modeling_ideogram4 as m  # noqa: E402

    return m


def _load_dit_gguf(model: torch.nn.Module, gguf_path: Path) -> None:
    import gguf

    reader = gguf.GGUFReader(str(gguf_path))
    state = dict(model.named_parameters())
    seen: set[str] = set()
    for t in reader.tensors:
        param = state.get(t.name)  # DiT GGUF keys are 1:1 with module paths.
        if param is None:
            raise KeyError(f"GGUF tensor {t.name!r} not in DiT state dict")
        flat = _dequantize(t)
        with torch.no_grad():
            param.copy_(torch.from_numpy(flat).to(param.dtype).reshape(param.shape))
        seen.add(t.name)
        del flat
    missing = set(state) - seen
    if missing:
        raise RuntimeError(f"DiT GGUF missing {len(missing)}: {sorted(missing)[:8]}")


def _fold_lora(model: torch.nn.Module, lora_path: Path) -> int:
    """`W += B@A` (scale alpha/rank = 1.0) at the 204 LoRA matmul sites."""
    from safetensors.torch import load_file

    lora = load_file(str(lora_path))
    params = dict(model.named_parameters())
    n_folded = 0
    n_layers = len(model.layers)
    for i in range(n_layers):
        for site in LORA_SITES:
            a_key = f"diffusion_model.layers.{i}.{site}.lora_A.weight"
            b_key = f"diffusion_model.layers.{i}.{site}.lora_B.weight"
            base_key = f"layers.{i}.{site}.weight"
            if a_key not in lora or b_key not in lora:
                continue
            base = params.get(base_key)
            if base is None:
                raise KeyError(f"LoRA base {base_key!r} not in DiT state dict")
            a = lora[a_key].to(torch.float32)  # [rank, K]
            b = lora[b_key].to(torch.float32)  # [N, rank]
            delta = (b @ a).reshape(base.shape)  # [N, K]
            with torch.no_grad():
                base.add_(delta.to(base.dtype))
            n_folded += 1
    print(f"[e2e-dit] folded {n_folded} LoRA sites", flush=True)
    return n_folded


def stage_dit(args) -> int:
    assert args.width % PATCH == 0 and args.height % PATCH == 0, "dims must be /16"
    grid_h = args.height // PATCH
    grid_w = args.width // PATCH
    num_image = grid_h * grid_w

    meta = (args.out / "meta_encode.txt").read_text().split()
    num_text = int(meta[0])
    llm_features_text = torch.from_numpy(
        np.fromfile(str(args.out / "llm_features.bin"), dtype="<f4")
    ).reshape(num_text, -1)
    seq = num_text + num_image

    m = _import_modeling()  # registers the stub `ideogram4` pkg for sibling imports
    from ideogram4.scheduler import get_schedule_for_resolution, make_step_intervals

    cfg = m.Ideogram4Config()
    torch.set_grad_enabled(False)

    torch.set_default_dtype(torch.bfloat16)
    try:
        model = m.Ideogram4Transformer(cfg).eval()
    finally:
        torch.set_default_dtype(torch.float32)
    print("[e2e-dit] loading DiT GGUF (dequant Q8 -> bf16)...", flush=True)
    _load_dit_gguf(model, args.dit_gguf)
    folded = _fold_lora(model, args.lora)
    assert folded == 6 * len(model.layers), f"expected {6 * len(model.layers)} folds, got {folded}"

    # Seeded noise the engine injects verbatim.
    gen = torch.Generator().manual_seed(args.seed)
    z = torch.randn(num_image, cfg.in_channels, dtype=torch.float32, generator=gen)
    _dump(z, args.out / "noise.bin")

    # Packed [text][image] inputs (B=1).
    feats = torch.zeros(1, seq, cfg.llm_features_dim, dtype=torch.float32)
    feats[0, :num_text, :] = llm_features_text
    indicator = torch.zeros(1, seq, dtype=torch.long)
    indicator[0, :num_text] = LLM_TOKEN_INDICATOR
    indicator[0, num_text:] = OUTPUT_IMAGE_INDICATOR
    segment_ids = torch.ones(1, seq, dtype=torch.long)
    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    for pi in range(num_text):
        position_ids[0, pi, :] = pi
    for r in range(grid_h):
        for c in range(grid_w):
            row = num_text + r * grid_w + c
            position_ids[0, row, 0] = IMAGE_POSITION_OFFSET
            position_ids[0, row, 1] = r + IMAGE_POSITION_OFFSET
            position_ids[0, row, 2] = c + IMAGE_POSITION_OFFSET

    schedule = get_schedule_for_resolution(
        (args.height, args.width), known_mean=args.mu, std=args.std
    )
    step_intervals = make_step_intervals(args.steps)
    text_z_padding = torch.zeros(1, num_text, cfg.in_channels, dtype=torch.float32)

    z = z.unsqueeze(0)  # [1, num_image, 128]
    for i in range(args.steps - 1, -1, -1):
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
        t = torch.full((1,), t_val, dtype=torch.float32)
        pos_z = torch.cat([text_z_padding, z], dim=1)
        out = model(
            llm_features=feats,
            x=pos_z,
            t=t,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )
        v = out[:, num_text:]
        z = z + v * (s_val - t_val)

    _dump(z[0], args.out / "latent.bin")
    (args.out / "meta_dit.txt").write_text(f"{num_image} {grid_h} {grid_w}\n")
    print(f"[e2e-dit] done: latent {tuple(z[0].shape)}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# stage: vae
# --------------------------------------------------------------------------- #
def stage_vae(args) -> int:
    from safetensors.torch import load_file

    _stub_ideogram4_pkg()
    from ideogram4 import autoencoder as ae  # noqa: E402
    from ideogram4 import latent_norm as ln  # noqa: E402

    meta = (args.out / "meta_dit.txt").read_text().split()
    num_image, grid_h, grid_w = int(meta[0]), int(meta[1]), int(meta[2])
    latent = torch.from_numpy(
        np.fromfile(str(args.out / "latent.bin"), dtype="<f4")
    ).reshape(1, num_image, 128)

    torch.set_grad_enabled(False)
    decoder = ae.Decoder(
        ch=96,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        in_channels=3,
        resolution=256,
        z_channels=32,
    ).eval()
    raw = load_file(str(args.vae))
    converted = ae.convert_diffusers_state_dict(raw)
    dec_state = {
        k[len("decoder.") :]: v for k, v in converted.items() if k.startswith("decoder.")
    }
    missing, unexpected = decoder.load_state_dict(dec_state, strict=False)
    assert not missing, f"decoder missing: {sorted(missing)[:8]}"
    assert not unexpected, f"decoder unexpected: {sorted(unexpected)[:8]}"

    shift, scale = ln.get_latent_norm()
    z = latent * scale + shift
    z = z.view(1, grid_h, grid_w, PATCH_SIZE, PATCH_SIZE, AE_CHANNELS)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(1, AE_CHANNELS, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)
    decoded = decoder(z.to(torch.float32)).clamp(-1.0, 1.0)  # [1,3,H,W] in [-1,1]
    _dump(decoded[0], args.out / "decoded.bin")
    print(
        f"[e2e-vae] done: decoded{tuple(decoded.shape)} "
        f"range[{decoded.min():.3f},{decoded.max():.3f}]",
        flush=True,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["encode", "dit", "vae"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--enc-gguf", type=Path)
    p.add_argument("--dit-gguf", type=Path)
    p.add_argument("--lora", type=Path)
    p.add_argument("--vae", type=Path)
    p.add_argument("--prompt", default="a red apple on a wooden table")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--mu", type=float, default=0.5)
    p.add_argument("--std", type=float, default=1.75)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "encode":
        assert args.enc_gguf, "--enc-gguf required for stage encode"
        return stage_encode(args)
    if args.stage == "dit":
        assert args.dit_gguf and args.lora, "--dit-gguf and --lora required for stage dit"
        return stage_dit(args)
    assert args.vae, "--vae required for stage vae"
    return stage_vae(args)


if __name__ == "__main__":
    raise SystemExit(main())
