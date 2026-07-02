"""HunyuanVideo 1.5 T2V DiT reference (full transformer, lightx2v 4-step).

Builds the T2V forward path from the cloned reference submodules + checkpoint
weights (the SAME fp16 bytes the engine loads, exact in f32) and runs it at tiny
dims. The dual-stream block math is reimplemented inline (faithful to
`MMDoubleStreamBlock.forward`) so we avoid the heavy `parallel_attention` /
flash-attn import path; the refiner (`txt_in`) reuses the upstream
`SingleTokenRefiner`, and RoPE / modulate / gate reuse the upstream helpers.

Dumps (LE f32):
  text_in.bin    [seq, 3584]    text hidden (engine input)
  img_tokens.bin [THW, 65]      [noise32 | 0 | 0] in (t,h,w) order (engine input)
  vec.bin        [2048]         time_in(t)
  img_in.bin     [THW, 2048]    img_in(hidden)
  txt_in.bin     [seq, 2048]    SingleTokenRefiner + cond_type[0]
  block0_img.bin [THW, 2048]    img stream after double block 0
  block0_txt.bin [seq, 2048]    txt stream after double block 0
  velocity.bin   [THW, 32]      final_layer output (pre-unpatchify == engine out)
  meta.txt       "seq T H W"

  uv run --with einops --with loguru python -m \\
      thinfer_pytorch_ref.hunyuan.gen_dit_ref \\
      --dit <...hy1.5_t2v_480p_lightx2v_4step.safetensors> --out <dir> \\
      [--seq 16 --t 4 --h 4 --w 4]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IN_CHANNELS = 3584
HIDDEN = 2048
HEADS = 16
HEAD_DIM = 128
DOUBLE_BLOCKS = 54
LATENT = 32
CONV_IN = 65
ROPE_DIM = [16, 56, 56]
ROPE_THETA = 256
TIMESTEP = 500.0


def _locate_hyvideo() -> str:
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "third-party" / "HunyuanVideo-1.5"
        if (cand / "hyvideo").is_dir():
            return str(cand)
    raise AssertionError("could not locate third-party/HunyuanVideo-1.5 above this script")


def _read_safetensors(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        out: dict[str, torch.Tensor] = {}
        dtype_map = {"F32": np.float32, "F16": np.float16}
        for name, info in header.items():
            if name == "__metadata__" or not name.startswith(prefix):
                continue
            a, b = info["data_offsets"]
            fh.seek(base + a)
            raw = fh.read(b - a)
            if info["dtype"] == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                f32 = (u16 << 16).view(np.float32)
                t = torch.from_numpy(f32.copy()).reshape(info["shape"])
            else:
                ndt = dtype_map[info["dtype"]]
                t = torch.from_numpy(np.frombuffer(raw, dtype=ndt).copy()).reshape(info["shape"])
                t = t.to(torch.float32)
            out[name[len(prefix):]] = t
    return out


def _layernorm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (HIDDEN,), eps=1e-6)


def _rms_head(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # x [B, L, H, D]; normalize over D, affine weight [D].
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return out * w


def load_dit(dit_path: Path):
    """Load the DiT weights + a fitted `SingleTokenRefiner` + the reference
    helper fns. Returns `(sd, refiner, mods)` reusable across forwards/steps."""
    sys.path.insert(0, _locate_hyvideo())
    from hyvideo.models.transformers.modules.token_refiner import SingleTokenRefiner
    from hyvideo.models.transformers.modules.embed_layers import timestep_embedding
    from hyvideo.models.transformers.modules.modulate_layers import modulate, apply_gate
    from hyvideo.models.transformers.modules.posemb_layers import (
        apply_rotary_emb,
        get_nd_rotary_pos_embed,
    )

    sd = _read_safetensors(dit_path, "")
    refiner = (
        SingleTokenRefiner(in_channels=IN_CHANNELS, hidden_size=HIDDEN, heads_num=HEADS, depth=2)
        .eval()
        .float()
    )
    rsd = {k[len("txt_in."):]: v for k, v in sd.items() if k.startswith("txt_in.")}
    miss, unexp = refiner.load_state_dict(rsd, strict=False)
    assert not miss, f"refiner missing: {miss}"
    assert not unexp, f"refiner unexpected: {unexp}"
    mods = dict(
        timestep_embedding=timestep_embedding,
        modulate=modulate,
        apply_gate=apply_gate,
        apply_rotary_emb=apply_rotary_emb,
        get_nd_rotary_pos_embed=get_nd_rotary_pos_embed,
    )
    return sd, refiner, mods


def dit_forward(sd, refiner, mods, text, hidden, t, grid, taps=None):
    """One T2V forward. `text [1,seq,3584]`, `hidden [1,65,T,H,W]`, `t [1]`,
    `grid=(T,H,W)`. Returns velocity `[1, THW, 32]`. Fills `taps` dict if given."""
    modulate = mods["modulate"]
    apply_gate = mods["apply_gate"]
    apply_rotary_emb = mods["apply_rotary_emb"]
    T, H, W = grid
    thw = T * H * W
    seq = text.shape[1]
    with torch.no_grad():
        txt = refiner(text, t, mask=None) + sd["cond_type_embedding.weight"][0]
        img = F.conv3d(hidden, sd["img_in.proj.weight"], sd["img_in.proj.bias"])
        img = img.flatten(2).transpose(1, 2)
        tfreq = mods["timestep_embedding"](t, 256, 10000)
        vec = F.linear(tfreq, sd["time_in.mlp.0.weight"], sd["time_in.mlp.0.bias"])
        vec = F.silu(vec)
        vec = F.linear(vec, sd["time_in.mlp.2.weight"], sd["time_in.mlp.2.bias"])
        cos, sin = mods["get_nd_rotary_pos_embed"](
            tuple(ROPE_DIM), (T, H, W), theta=ROPE_THETA, use_real=True, theta_rescale_factor=1
        )
        freqs_cis = (cos, sin)
        if taps is not None:
            taps["vec"] = vec.clone()
            taps["img_in"] = img.clone()
            taps["txt_in"] = txt.clone()

        for i in range(DOUBLE_BLOCKS):
            pre = f"double_blocks.{i}."

            def lin(name, x):
                return F.linear(x, sd[pre + name + ".weight"], sd[pre + name + ".bias"])

            imod = F.linear(F.silu(vec), sd[pre + "img_mod.linear.weight"], sd[pre + "img_mod.linear.bias"])
            tmod = F.linear(F.silu(vec), sd[pre + "txt_mod.linear.weight"], sd[pre + "txt_mod.linear.bias"])
            i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = imod.chunk(6, dim=-1)
            t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = tmod.chunk(6, dim=-1)

            im = modulate(_layernorm(img), shift=i_sh1, scale=i_sc1)
            iq = lin("img_attn_q", im).reshape(1, thw, HEADS, HEAD_DIM)
            ik = lin("img_attn_k", im).reshape(1, thw, HEADS, HEAD_DIM)
            iv = lin("img_attn_v", im).reshape(1, thw, HEADS, HEAD_DIM)
            iq = _rms_head(iq, sd[pre + "img_attn_q_norm.weight"])
            ik = _rms_head(ik, sd[pre + "img_attn_k_norm.weight"])
            iq, ik = apply_rotary_emb(iq, ik, freqs_cis, head_first=False)

            tm = modulate(_layernorm(txt), shift=t_sh1, scale=t_sc1)
            tq = lin("txt_attn_q", tm).reshape(1, seq, HEADS, HEAD_DIM)
            tk = lin("txt_attn_k", tm).reshape(1, seq, HEADS, HEAD_DIM)
            tv = lin("txt_attn_v", tm).reshape(1, seq, HEADS, HEAD_DIM)
            tq = _rms_head(tq, sd[pre + "txt_attn_q_norm.weight"])
            tk = _rms_head(tk, sd[pre + "txt_attn_k_norm.weight"])

            q = torch.cat([iq, tq], dim=1)
            k = torch.cat([ik, tk], dim=1)
            v = torch.cat([iv, tv], dim=1)
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
            x = x.reshape(1, thw + seq, HIDDEN)
            img_attn, txt_attn = x[:, :thw], x[:, thw:]

            img = img + apply_gate(lin("img_attn_proj", img_attn), gate=i_g1)
            im2 = modulate(_layernorm(img), shift=i_sh2, scale=i_sc2)
            img_mlp = F.linear(
                F.gelu(lin("img_mlp.fc1", im2), approximate="tanh"),
                sd[pre + "img_mlp.fc2.weight"],
                sd[pre + "img_mlp.fc2.bias"],
            )
            img = img + apply_gate(img_mlp, gate=i_g2)

            txt = txt + apply_gate(lin("txt_attn_proj", txt_attn), gate=t_g1)
            tm2 = modulate(_layernorm(txt), shift=t_sh2, scale=t_sc2)
            txt_mlp = F.linear(
                F.gelu(lin("txt_mlp.fc1", tm2), approximate="tanh"),
                sd[pre + "txt_mlp.fc2.weight"],
                sd[pre + "txt_mlp.fc2.bias"],
            )
            txt = txt + apply_gate(txt_mlp, gate=t_g2)

            if i == 0 and taps is not None:
                taps["block0_img"] = img.clone()
                taps["block0_txt"] = txt.clone()

        emb = F.linear(
            F.silu(vec),
            sd["final_layer.adaLN_modulation.1.weight"],
            sd["final_layer.adaLN_modulation.1.bias"],
        )
        shift, scale = emb.chunk(2, dim=-1)
        out = modulate(_layernorm(img), shift=shift, scale=scale)
        return F.linear(out, sd["final_layer.linear.weight"], sd["final_layer.linear.bias"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dit", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seq", type=int, default=16)
    p.add_argument("--t", type=int, default=4)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--w", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sd, refiner, mods = load_dit(args.dit)
    seq, T, H, W = args.seq, args.t, args.h, args.w
    thw = T * H * W
    g = torch.Generator().manual_seed(args.seed)

    text = torch.randn(1, seq, IN_CHANNELS, generator=g, dtype=torch.float32)
    noise = torch.randn(1, LATENT, T, H, W, generator=g, dtype=torch.float32)
    hidden = torch.cat(
        [noise, torch.zeros(1, LATENT, T, H, W), torch.zeros(1, 1, T, H, W)], dim=1
    )  # [1, 65, T, H, W]
    t = torch.tensor([TIMESTEP], dtype=torch.float32)

    taps: dict[str, torch.Tensor] = {}
    velocity = dit_forward(sd, refiner, mods, text, hidden, t, (T, H, W), taps=taps)

    def dump(name: str, tt: torch.Tensor) -> None:
        arr = tt.detach().contiguous().to(torch.float32).numpy().astype("<f4")
        (args.out / name).write_bytes(arr.tobytes())

    img_tokens = hidden[0].permute(1, 2, 3, 0).reshape(thw, CONV_IN)
    dump("text_in.bin", text[0])
    dump("img_tokens.bin", img_tokens)
    dump("vec.bin", taps["vec"][0])
    dump("img_in.bin", taps["img_in"][0])
    dump("txt_in.bin", taps["txt_in"][0])
    dump("block0_img.bin", taps["block0_img"][0])
    dump("block0_txt.bin", taps["block0_txt"][0])
    dump("velocity.bin", velocity[0])
    (args.out / "meta.txt").write_text(f"{seq} {T} {H} {W}\n")
    print(f"hunyuan dit ref: seq={seq} grid=({T},{H},{W}) -> velocity[{thw},{LATENT}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
