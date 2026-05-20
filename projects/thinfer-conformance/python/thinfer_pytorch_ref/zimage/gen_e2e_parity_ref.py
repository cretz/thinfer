"""End-to-end Z-Image reference: prompt + pinned noise -> RGB pixels.

This is the FOREVER parity reference. Unlike `gen_dit_parity_ref`, it does
NOT install any DiT-internal hooks (no nr0/nr1 re-implementation, no
t_embedder/cap_embedder taps, no transformer post-hooks). The pipeline runs
through to completion, including VAE decode. Only externally-observable
stages are captured:

  - prepare_latents.out (== starting latents the scheduler sees)
  - scheduler.step prev_sample, per step
  - vae.decode input (pre-VAE latent) and output (CHW fp32 RGB in [-1, 1])

The Rust test side mirrors this exactly via `model.denoise_with(..,
step_dumps=Some(..))` followed by `model.vae.decode(..)`. If every stage
matches, end-to-end byte parity is established. If a stage diverges, the
narrower `dit_parity` / `qwen3_parity` tests localize it.

Usage:

    uv run python -m thinfer_pytorch_ref.zimage.gen_e2e_parity_ref \\
        --initial-noise <noise.bin> \\
        --transformer-shard <shard1.safetensors> \\
        --transformer-shard <shard2.safetensors> \\
        --prompt "..." --height 256 --width 256 --steps 2 --seed 42 \\
        --out <tmpdir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DiffusionPipeline
from safetensors.torch import load_file as load_safetensors


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

REPO = "Tongyi-MAI/Z-Image-Turbo"
LATENT_CHANNELS = 16
VAE_SCALE = 8


def _dump(t: torch.Tensor, path: Path) -> None:
    arr = t.detach().to(torch.float32).cpu().numpy().astype("<f4")
    arr.tofile(str(path))


def _summarize(label: str, t: torch.Tensor) -> None:
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    print(
        f"  [PY-DUMP] {label}: len={a.size} min={a.min():.5e} "
        f"max={a.max():.5e} max_abs={abs(a).max():.5e} mean={a.mean():.5e}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-noise", required=True, type=Path)
    p.add_argument(
        "--transformer-shard",
        required=True,
        type=Path,
        action="append",
        help="Safetensors file(s) holding the DiT state dict. Repeat for shards.",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory. All per-stage .bin files written as siblings.",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--dtype", choices=list(DTYPES.keys()), default="bf16")
    p.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="If set, save the py PNG here (filename via --png-filename).",
    )
    p.add_argument(
        "--png-filename",
        default="py.png",
        help=(
            "Filename for the dumped py PNG inside --png-dir. The rust "
            "side stamps the variant slug here (e.g. py_safetensors.png "
            "/ py_gguf_q8_0.png) so multiple e2e variants can share the "
            "same png dir."
        ),
    )
    p.add_argument(
        "--vae-diag-dir",
        type=Path,
        default=None,
        help=(
            "If set, hook every pipe.vae.decoder submodule and dump the "
            "first 256 fp32 elements of each output to py_<stage>.bin in "
            "this directory. Matches the rust side's ours-side dumps so "
            "the test can per-stage compare."
        ),
    )
    args = p.parse_args()

    if args.height % (VAE_SCALE * 2) or args.width % (VAE_SCALE * 2):
        raise SystemExit(
            f"height/width must be multiples of {VAE_SCALE * 2}; "
            f"got {args.height}x{args.width}"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    print(f"device={device} dtype={dtype}", flush=True)

    h_lat = args.height // VAE_SCALE
    w_lat = args.width // VAE_SCALE
    n_expected = LATENT_CHANNELS * h_lat * w_lat
    raw = np.fromfile(str(args.initial_noise), dtype="<f4")
    if raw.size != n_expected:
        raise SystemExit(
            f"--initial-noise has {raw.size} fp32 values; expected "
            f"{n_expected} for [1, {LATENT_CHANNELS}, {h_lat}, {w_lat}]"
        )
    latents = (
        torch.from_numpy(raw.copy())
        .reshape(1, LATENT_CHANNELS, h_lat, w_lat)
        .to(dtype=dtype, device=device)
    )

    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        REPO, torch_dtype=dtype, local_files_only=True
    )
    print(
        f"loaded pipeline in {time.time() - t0:.1f}s "
        f"(class={type(pipe).__name__})",
        flush=True,
    )

    # Same transformer-override discipline as dit_parity: refuse silent
    # partial loads so any divergence is attributable to engine math, not
    # weight provenance.
    merged: dict[str, torch.Tensor] = {}
    for shard in args.transformer_shard:
        d = load_safetensors(str(shard))
        for k, v in d.items():
            if k in merged:
                raise SystemExit(f"duplicate key {k!r} across shards")
            merged[k] = v
    merged = {k: v.to(dtype) for k, v in merged.items()}
    missing, unexpected = pipe.transformer.load_state_dict(merged, strict=False)
    if missing or unexpected:
        print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        raise SystemExit(
            "transformer override key mismatch; refusing to run with "
            "partially-loaded weights"
        )
    print(f"transformer override: {len(merged)} keys", flush=True)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # --- hook prepare_latents: dump starting latents the scheduler sees ---
    starting_path = args.out / "py_starting_latents.bin"
    orig_prepare_latents = pipe.prepare_latents

    def hooked_prepare_latents(*pa, **kw):
        out = orig_prepare_latents(*pa, **kw)
        _summarize("prepare_latents.out (starting latents)", out)
        _dump(out, starting_path)
        return out

    pipe.prepare_latents = hooked_prepare_latents  # type: ignore[assignment]

    # --- hook scheduler.step: dump per-step prev_sample ---
    orig_scheduler_step = pipe.scheduler.step
    step_counter = {"i": 0}

    def hooked_scheduler_step(model_output, timestep, sample, *pa, **kw):
        i = step_counter["i"]
        sigma_idx = (
            pipe.scheduler.step_index if pipe.scheduler.step_index is not None else 0
        )
        sigma = float(pipe.scheduler.sigmas[sigma_idx].item())
        sigma_next = float(pipe.scheduler.sigmas[sigma_idx + 1].item())
        dt = sigma_next - sigma
        t_val = float(timestep.item()) if hasattr(timestep, "item") else float(timestep)
        print(
            f"  [PY-DUMP] scheduler.step i={i} t={t_val:.6f} "
            f"sigma={sigma:.6f} sigma_next={sigma_next:.6f} dt={dt:.6f}",
            flush=True,
        )
        _summarize(f"step{i}.model_output (post-negation, fed to scheduler)", model_output)
        _summarize(f"step{i}.sample_in", sample)
        result = orig_scheduler_step(model_output, timestep, sample, *pa, **kw)
        prev = result[0] if isinstance(result, tuple) else result.prev_sample
        _summarize(f"step{i}.prev_sample (post-step)", prev)
        _dump(prev, args.out / f"py_step{i}_post.bin")
        step_counter["i"] += 1
        return result

    pipe.scheduler.step = hooked_scheduler_step  # type: ignore[assignment]

    # --- hook vae.decode: capture pre-VAE latent AND post-VAE RGB ---
    # We dump the decoder INPUT (latent fed to vae.decode) and OUTPUT
    # (CHW fp32 RGB in [-1, 1], before any image_processor postprocess).
    # This is the byte-for-byte counterpart to our Rust `vae.decode` call.
    pre_vae_path = args.out / "py_pre_vae_latent.bin"
    vae_rgb_path = args.out / "py_vae_rgb.bin"
    orig_vae_decode = pipe.vae.decode

    def hooked_vae_decode(z, *pa, **kw):
        _summarize("vae.decode input (pre-VAE latent)", z)
        _dump(z, pre_vae_path)
        result = orig_vae_decode(z, *pa, **kw)
        # diffusers AutoencoderKL.decode returns DecoderOutput(sample=...).
        rgb = result.sample if hasattr(result, "sample") else result[0]
        _summarize("vae.decode output (RGB CHW in [-1, 1])", rgb)
        _dump(rgb, vae_rgb_path)
        return result

    pipe.vae.decode = hooked_vae_decode  # type: ignore[assignment]

    # --- opt-in: hook every pipe.vae.decoder submodule for per-stage diag.
    # Matches the rust side's stage labels from `decoder_back` (front_in,
    # up{i}.resnet{j}, up{i}.upconv, conv_norm_out, conv_out). Each hook
    # dumps the first 256 fp32 elements of the output - bounded to mirror
    # the rust side's `STAGE_DIAG_MAX_BYTES` cap (1 KiB), which exists
    # because MiB-scale VAE readbacks crash the wgpu device. ---
    if args.vae_diag_dir is not None:
        args.vae_diag_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale dumps so a stale file can't mask a missing hook fire.
        for f in args.vae_diag_dir.iterdir():
            if f.suffix == ".bin":
                f.unlink()
        HEAD_N = 256

        def _dump_head(t: torch.Tensor, label: str) -> None:
            arr = (
                t.detach()
                .contiguous()
                .to(torch.float32)
                .cpu()
                .numpy()
                .ravel()[:HEAD_N]
                .astype("<f4")
            )
            out_p = args.vae_diag_dir / f"py_{label}.bin"
            arr.tofile(str(out_p))

        def _mk_hook(label: str):
            def _hook(_m, _i, output):
                # diffusers VAE decoder modules return a raw Tensor (not a
                # namedtuple) at the submodule level; guard anyway.
                t = output[0] if isinstance(output, tuple) else output
                if isinstance(t, torch.Tensor):
                    _dump_head(t, label)
                return output
            return _hook

        dec = pipe.vae.decoder
        # `front_in` on our side is the input to decoder_back, which is
        # post-conv_in + post-mid_block. For the single-tile config that
        # matches mid_block's output byte-for-byte.
        dec.mid_block.register_forward_hook(_mk_hook("front_in"))
        for i, ub in enumerate(dec.up_blocks):
            for j, rn in enumerate(ub.resnets):
                rn.register_forward_hook(_mk_hook(f"up{i}.resnet{j}"))
            ups = getattr(ub, "upsamplers", None)
            if ups is not None:
                # `up{i}.upconv` on our side = post-(nearest upsample +
                # conv); diffusers' Upsample2D rolls both into one module
                # output, so we hook the combined module.
                ups[0].register_forward_hook(_mk_hook(f"up{i}.upconv"))
        dec.conv_norm_out.register_forward_hook(_mk_hook("conv_norm_out"))
        dec.conv_out.register_forward_hook(_mk_hook("conv_out"))
        print(
            f"  [vae-diag] hooks installed on {1 + sum(len(ub.resnets) for ub in dec.up_blocks)} resnets/mid + "
            f"{sum(1 for ub in dec.up_blocks if getattr(ub, 'upsamplers', None))} upsamplers + conv_norm_out + conv_out",
            flush=True,
        )

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.time()
    # output_type="pt" returns a tensor we can dump cleanly. guidance_scale=0
    # matches Z-Image-Turbo (no CFG; positive prompt only, bsz=1).
    out = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=gen,
        latents=latents,
        guidance_scale=0.0,
        output_type="pt",
    )
    print(f"denoised + decoded in {time.time() - t1:.1f}s", flush=True)

    # Post-processed image: [B, C, H, W] float in [0, 1] (or u8 depending on
    # version). Dump as fp32 so the rust side can compare against its own
    # (rgb+1)/2 transformation.
    img = out.images if hasattr(out, "images") else out[0]
    if isinstance(img, list):
        img = img[0]
    if isinstance(img, torch.Tensor):
        _summarize("final image (post-processor)", img)
        _dump(img.float(), args.out / "py_final_image.bin")

    if args.png_dir is not None:
        from PIL import Image
        args.png_dir.mkdir(parents=True, exist_ok=True)
        # ONE png per side. Use raw VAE output transformed with the same
        # (v+1)*127.5 formula our rust `encode_png` uses, so py.png and
        # ours.png are byte-comparable when engine math matches. (The
        # image_processor.postprocess path would be a different conversion
        # and obscures any apples-to-apples diff.)
        raw_path = args.out / "py_vae_rgb.bin"
        if raw_path.exists():
            raw = np.fromfile(str(raw_path), dtype="<f4")
            c, h, w = 3, args.height, args.width
            if raw.size == c * h * w:
                chw = raw.reshape(c, h, w)
                interleaved = ((np.clip(chw, -1.0, 1.0) + 1.0) * 127.5).round().astype("uint8")
                interleaved = np.transpose(interleaved, (1, 2, 0))
                out_png = args.png_dir / args.png_filename
                Image.fromarray(interleaved).save(str(out_png))
                print(f"wrote {out_png}", flush=True)

    for p_ in (starting_path, pre_vae_path, vae_rgb_path):
        if p_.exists():
            print(f"wrote {p_} ({p_.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
