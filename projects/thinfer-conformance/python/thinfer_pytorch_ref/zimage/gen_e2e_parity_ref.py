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
        f"max={a.max():.5e} max_abs={abs(a).max():.5e} mean={a.mean():.5e} "
        f"mean_abs={abs(a).mean():.5e}",
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

    # --- DIAG: hook transformer.layers[0] (main DiT block 0) attention to
    # capture sdpa output (input to to_out[0]) and projection output
    # (output of to_out[0]) so the I8-acts engine can A/B against these.
    # Fires once per step; we print the first step's numbers.
    block0 = pipe.transformer.layers[0]
    diag_state = {"sdpa_in_printed": False, "attn_out_printed": False, "block_in_printed": False}

    def _to_out_pre_hook(_m, args_in):
        if diag_state["sdpa_in_printed"]:
            return
        diag_state["sdpa_in_printed"] = True
        x = args_in[0]
        _summarize("PYDIAG block0.attn_sdpa (input to to_out[0])", x)

    def _to_out_post_hook(_m, _i, output):
        if diag_state["attn_out_printed"]:
            return
        diag_state["attn_out_printed"] = True
        _summarize("PYDIAG block0.attn_out (output of to_out[0])", output)

    def _block_pre_hook(_m, args_in):
        if diag_state["block_in_printed"]:
            return
        diag_state["block_in_printed"] = True
        _summarize("PYDIAG block0.input (x before block forward)", args_in[0])

    block0.attention.to_out[0].register_forward_pre_hook(_to_out_pre_hook)
    block0.attention.to_out[0].register_forward_hook(_to_out_post_hook)
    block0.register_forward_pre_hook(_block_pre_hook)

    # --- DIAG: per-stage probes matching engine-side `[DIT-PROBE] NN_xxx`
    # labels so the two sides can be A/B'd by grepping the same suffix.
    # Each hook fires once (first step only) to keep log volume bounded.
    pydiag_fired: dict[str, bool] = {}

    # DIAG: every PYDIAG label now also writes a `py_<label>.bin` so the
    # rust side can per-cell linfit it. Volume is bounded (each label
    # fires once on step 0, biggest tensors are seq_u*dim=1.1M f32 ~ 4MB).
    def _mk_once_post(label: str):
        def _hook(_m, _inp, output):
            if pydiag_fired.get(label):
                return
            pydiag_fired[label] = True
            t = output[0] if isinstance(output, tuple) else output
            if isinstance(t, torch.Tensor):
                _summarize(f"PYDIAG {label}", t)
                _dump(t, args.out / f"py_{label}.bin")

        return _hook

    def _mk_once_pre(label: str, arg_idx: int = 0):
        def _hook(_m, args_in):
            if pydiag_fired.get(label):
                return
            pydiag_fired[label] = True
            t = args_in[arg_idx]
            if isinstance(t, torch.Tensor):
                _summarize(f"PYDIAG {label}", t)
                _dump(t, args.out / f"py_{label}.bin")

        return _hook

    xf = pipe.transformer
    if hasattr(xf, "x_embedder"):
        xf.x_embedder.register_forward_hook(_mk_once_post("01_x_embedder_out"))
    if hasattr(xf, "cap_embedder"):
        xf.cap_embedder.register_forward_hook(_mk_once_post("05_cap_embedder_out"))
    if hasattr(xf, "noise_refiner"):
        for i, blk in enumerate(xf.noise_refiner):
            blk.register_forward_hook(_mk_once_post(f"04_noise_refiner_block{i}_out"))
    if hasattr(xf, "context_refiner"):
        for i, blk in enumerate(xf.context_refiner):
            blk.register_forward_hook(_mk_once_post(f"07_context_refiner_block{i}_out"))
    if hasattr(xf, "layers"):
        # 08_unified_in == input to layers[0] (i.e. the concatenated x;cap).
        xf.layers[0].register_forward_pre_hook(_mk_once_pre("08_unified_in_pre_main_block0"))
        # Every main block residual output (post-block, full residual).
        # 30 dumps + linfit at each gives a per-block slope curve; the
        # first block where slope deviates from 1.0 localizes the bug.
        for n in range(len(xf.layers)):
            xf.layers[n].register_forward_hook(_mk_once_post(f"09_main_block{n}_out"))
        # Intra-block per-op dumps at block 0 (start), block 14 (mid)
        # and blocks 25-29 (already bisecting). Block 0 and block 29
        # are the new ones; gives us slope-per-op at start AND end of
        # the stack so we can tell whether per-op shrink is uniform
        # (compounding bug) or grows with block index (data-dependent).
        for n in (0, 14, 25, 26, 27, 28, 29):
            if n < len(xf.layers):
                blk = xf.layers[n]
                blk.attention.register_forward_hook(
                    _mk_once_post(f"09_main_block{n}_attn_out")
                )
                blk.feed_forward.register_forward_hook(
                    _mk_once_post(f"09_main_block{n}_ffn_out")
                )
                if hasattr(blk, "adaLN_modulation"):
                    blk.adaLN_modulation.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_adaln_mod")
                    )
                # Intra-attention bisect: modulated_attn_in (pre), to_q/k/v
                # outputs (post-projection, pre-rope/sdpa), to_out[0] (post
                # o_proj — equals attn_out, included for sanity).
                blk.attention.register_forward_pre_hook(
                    _mk_once_pre(f"09_main_block{n}_modulated_attn_in")
                )
                attn = blk.attention
                if hasattr(attn, "to_q"):
                    attn.to_q.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_to_q")
                    )
                if hasattr(attn, "to_k"):
                    attn.to_k.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_to_k")
                    )
                if hasattr(attn, "to_v"):
                    attn.to_v.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_to_v")
                    )
                if hasattr(attn, "to_out"):
                    attn.to_out[0].register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_to_out0")
                    )
                    # to_out[0] is the o_proj Linear; its INPUT is the
                    # post-sdpa (head-merged) tensor. Capturing it lets us
                    # compare py's sdpa output against ours attn_sdpa and
                    # decide whether the attn_out bias originates in sdpa
                    # vs. in the o_proj matmul_i8.
                    attn.to_out[0].register_forward_pre_hook(
                        _mk_once_pre(f"09_main_block{n}_to_out0_in")
                    )
                # RMSNorm probes to mirror our attn_norm2_out / ffn_norm1_out
                # / ffn_norm2_out taps. Matches the residual-path values we
                # already capture on the I8 side.
                if hasattr(blk, "attention_norm2"):
                    blk.attention_norm2.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_attn_norm2_out")
                    )
                if hasattr(blk, "ffn_norm1"):
                    blk.ffn_norm1.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_ffn_norm1_out")
                    )
                    # ffn_norm1's INPUT is x_mid (the post-attention residual
                    # add: x + gate_msa * attn_norm2(attn_out)). Capturing it
                    # isolates the residual-add step in the per-op chain --
                    # the i8 engine re-quantizes the sum there.
                    blk.ffn_norm1.register_forward_pre_hook(
                        _mk_once_pre(f"09_main_block{n}_x_mid")
                    )
                if hasattr(blk, "ffn_norm2"):
                    blk.ffn_norm2.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_ffn_norm2_out")
                    )
                # Intra-FFN: modulated_ffn_in (pre), w1/w3 (parallel gate +
                # up projections), w2 is the final output and == ffn_out.
                blk.feed_forward.register_forward_pre_hook(
                    _mk_once_pre(f"09_main_block{n}_modulated_ffn_in")
                )
                ff = blk.feed_forward
                if hasattr(ff, "w1"):
                    ff.w1.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_ffn_w1")
                    )
                if hasattr(ff, "w3"):
                    ff.w3.register_forward_hook(
                        _mk_once_post(f"09_main_block{n}_ffn_w3")
                    )
        last = len(xf.layers) - 1
        xf.layers[last].register_forward_hook(_mk_once_post("10_main_block29_out_post_drain"))

    # DIAG: dump the FULL last-main-block residual and final_layer_out as
    # binary tensors so the rust side can per-cell linfit them. Without
    # this, slope-vs-pyref can only be computed at the scheduler output;
    # we want to localize whether the slope appears in block 29's
    # residual or only after final_layer.
    def _mk_once_post_dump(label: str):
        fired = {"v": False}
        def _hook(_m, _inp, output):
            if fired["v"]:
                return
            fired["v"] = True
            t = output[0] if isinstance(output, tuple) else output
            _summarize(f"PYDIAG {label}", t)
            _dump(t, args.out / f"py_{label}.bin")
        return _hook

    if hasattr(xf, "layers") and len(xf.layers) > 0:
        last = len(xf.layers) - 1
        xf.layers[last].register_forward_hook(
            _mk_once_post_dump("10_main_block29_full")
        )
    if hasattr(xf, "final_layer"):
        xf.final_layer.register_forward_hook(
            _mk_once_post_dump("11_final_layer_full")
        )
    elif hasattr(xf, "all_final_layer"):
        for k, m in xf.all_final_layer.items():
            m.register_forward_hook(_mk_once_post_dump("11_final_layer_full"))
    if hasattr(xf, "final_layer"):
        xf.final_layer.register_forward_hook(_mk_once_post("11_final_layer_out"))
    elif hasattr(xf, "all_final_layer"):
        # Z-Image diffusers exposes `all_final_layer` as a ModuleDict keyed by
        # resolution-bucket; register on every entry so whichever one fires
        # produces the probe.
        for k, m in xf.all_final_layer.items():
            m.register_forward_hook(_mk_once_post("11_final_layer_out"))

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
