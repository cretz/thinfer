"""Ideogram-4 VAE-decode parity reference.

Loads the Flux2 KL autoencoder DECODER from the FLUX.2-VAE safetensors (the
ungated VAE the engine loads), generates a deterministic packed latent
`[num_image, 128]`, applies the `Ideogram4Pipeline._decode` denorm + unpatch,
runs the decoder, and dumps:

  latent_tokens.bin  f32 LE [num_image, 128]   packed latent (DiT-output shape).
  decoded.bin        f32 LE [3, H, W]          decoder RGB in [-1, 1] (pre-uint8).
  meta.txt           "num_image grid_h grid_w"

The engine reads latent_tokens.bin, runs the SAME host denorm+unpatch
(`ideogram4::vae::unpatch_denorm`), and decodes with the shared KL-VAE decoder;
both sides consume identical FLUX.2-VAE weights so the band is just the
engine's f16-act / bf16-weight rounding vs this fp32 reference.

The FLUX.2-VAE decoder is `ch=96` (mid 384, final 96), z_channels 32, with a
1x1 post_quant_conv -- NOT the `AutoEncoderParams` default (ch=128). We build
the decoder with the matching params and load only the `decoder.*` weights.

Memory: the VAE is ~0.16B; fp32 CPU at tiny dims (64x64) is trivial.

Usage:

    uv run --with einops python -m thinfer_pytorch_ref.ideogram4.gen_vae_ref \\
        --vae <FLUX.2-VAE diffusion_pytorch_model.safetensors> --out <dir> \\
        --width 64 --height 64
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file

PATCH = 16  # patch_size(2) * ae_scale_factor(8)
AE_CHANNELS = 32
PATCH_SIZE = 2


def _find_third_party() -> Path:
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "third-party" / "ideogram4" / "src"
        if cand.exists():
            return cand
    raise FileNotFoundError("third-party/ideogram4/src not found above this file")


def _import_ideogram4():
    """Register a stub `ideogram4` package so the lightweight submodules
    (`autoencoder`, `latent_norm`) import without the heavy real `__init__`."""
    src = _find_third_party()
    if not (src / "ideogram4" / "autoencoder.py").exists():
        raise FileNotFoundError(f"third-party ideogram4 not found at {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if "ideogram4" not in sys.modules:
        pkg = types.ModuleType("ideogram4")
        pkg.__path__ = [str(src / "ideogram4")]
        sys.modules["ideogram4"] = pkg
    from ideogram4 import autoencoder as ae  # noqa: E402
    from ideogram4 import latent_norm as ln  # noqa: E402

    return ae, ln


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assert args.width % PATCH == 0 and args.height % PATCH == 0, "dims must be /16"
    grid_h = args.height // PATCH
    grid_w = args.width // PATCH
    num_image = grid_h * grid_w

    ae, ln = _import_ideogram4()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    # Decoder built to the ACTUAL FLUX.2-VAE decoder widths (ch=96), not the
    # autoencoder.py default. resolution only sizes an unused buffer.
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
        k[len("decoder.") :]: v
        for k, v in converted.items()
        if k.startswith("decoder.")
    }
    missing, unexpected = decoder.load_state_dict(dec_state, strict=False)
    # `up.*.attn` ModuleLists are empty here (no per-up attention) so there are
    # no spurious missing/unexpected decoder params; assert a clean load.
    assert not missing, f"decoder missing params: {sorted(missing)[:8]}"
    assert not unexpected, f"decoder unexpected params: {sorted(unexpected)[:8]}"

    # Deterministic packed latent [1, num_image, 128] (DiT-output shape).
    latent = torch.randn(1, num_image, 128, dtype=torch.float32)
    _dump(latent[0], args.out / "latent_tokens.bin")

    # _decode: per-channel denorm over the 128-dim patch, then unpatch.
    shift, scale = ln.get_latent_norm()  # both [128] f32
    z = latent * scale + shift
    ae_channels = z.shape[-1] // (PATCH_SIZE * PATCH_SIZE)
    assert ae_channels == AE_CHANNELS
    z = z.view(1, grid_h, grid_w, PATCH_SIZE, PATCH_SIZE, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(1, ae_channels, grid_h * PATCH_SIZE, grid_w * PATCH_SIZE)

    decoded = decoder(z.to(torch.float32))  # [1, 3, H, W] in ~[-1, 1]
    _dump(decoded[0], args.out / "decoded.bin")
    (args.out / "meta.txt").write_text(f"{num_image} {grid_h} {grid_w}\n")
    print(
        f"[py-ideo-vae] done: num_image={num_image} grid={grid_h}x{grid_w} "
        f"decoded{tuple(decoded.shape)} range[{decoded.min():.3f},{decoded.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
