"""End-to-end SkyReels-V2-DF (Wan) video reference: prompt + pinned noise ->
RGB frames. The FOREVER parity reference for `tests/wan/video_e2e_parity.rs`.

The pipeline runs to completion, including VAE decode, with best-effort hooks
that capture every stage the Rust side also taps, so a per-stage byte/linfit
compare localizes any divergence to umT5 / DiT-velocity / scheduler / a specific
DiT block / VAE. The hooks never alter the math (they read + dump only) and any
signature mismatch prints + skips rather than crashing the reference. Dumps:

  - umt5_out             prompt_embeds [1, 512, 4096] (zero-padded to 512, no mask)
  - umt5_* (per-op)      embeds / per-block / per-layer-op (the ~0.365x scale bug)
  - dit_out_step{i}      raw transformer output (flow velocity) per step
  - block{i}_out_step{s} every DiT block's output residual, EVERY step
  - temb/timestep_proj/proj_out   within-DiT condition + head taps
  - step{i}_post         per-step latent after scheduler.step (== our_step{i})
  - pre_vae_latent       final denoised latent, PRE mean/std prescale (== our_pre_vae)
  - vae_rgb              vae.decode output [3, F, H, W] in [-1, 1] (== our_frames)

Pinned to the Rust config (wan-plan "e2e parity"): 64x64, F=5, steps 2, seed 42,
synchronous DF (ar_step=0, causal_block_size left None -> num_frame_per_block=1
-> full attention, no causal mask). fps=24 -> fps bucket 1 (the pipeline maps
`0 if fps==16 else 1`), matching the Rust `FPS_BUCKET=1`.

Two SkyReels-pipeline quirks this script works around:
  - `prepare_latents` early-returns a bare tensor when `latents is not None`,
    but the caller unpacks a 4-tuple -> passing `latents=` to `__call__` crashes.
    We inject the pinned noise by wrapping `prepare_latents` (run the random
    path for the correct tuple, then substitute our noise) instead.
  - guidance_scale 1.0 => do_classifier_free_guidance is False => one DiT
    forward per step, no negative prompt. Matches the engine's no-CFG path.

PNG staging mirrors the Rust `stage_frames`: a per-frame `<prefix>_frame{n}.png`
sequence plus a near-square `<prefix>_contact.png` contact sheet (black/-1.0
background), pixels via `((clip(v,-1,1)+1)*127.5).round()` so `py_*` and `ours_*`
are visually comparable when the engine math matches.

Usage:

    uv run python -m thinfer_pytorch_ref.wan.gen_video_e2e_parity_ref \\
        --initial-noise <noise.bin (f32 LE, 16*f_lat*h_lat*w_lat)> \\
        --out <tmpdir> --prompt "..." \\
        --height 64 --width 64 --num-frames 5 --steps 2 --seed 42 --fps 24 \\
        [--png-dir <dir> --png-prefix py]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

LATENT_CHANNELS = 16
VAE_SCALE = 8
TEMPORAL_SCALE = 4


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _summarize(label: str, t: torch.Tensor) -> None:
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    nan = int(np.isnan(a).sum())
    finite = a[np.isfinite(a)]
    lo = float(finite.min()) if finite.size else float("nan")
    hi = float(finite.max()) if finite.size else float("nan")
    ma = float(np.abs(finite).max()) if finite.size else float("nan")
    mean = float(finite.mean()) if finite.size else float("nan")
    print(
        f"  [PY-DUMP] {label}: len={a.size} nan={nan} min={lo:.5e} max={hi:.5e} "
        f"max_abs={ma:.5e} mean={mean:.5e}",
        flush=True,
    )


def _stage_frames(rgb_cthw: np.ndarray, n: int, h: int, w: int, png_dir: Path, prefix: str) -> None:
    """Replicate the Rust `stage_frames`: per-frame PNGs + a near-square contact
    sheet, both from CTHW frames in [-1, 1]. `rgb_cthw` is [3, N, H, W]."""
    from PIL import Image

    png_dir.mkdir(parents=True, exist_ok=True)
    per = h * w

    def frame_chw(f: int) -> np.ndarray:
        # Gather channel-planar [3, H, W] for frame f out of [3, N, H, W].
        return rgb_cthw[:, f, :, :].reshape(3, h, w)

    def to_u8_hwc(chw: np.ndarray) -> np.ndarray:
        # Match Rust encode_png: ((clip(v,-1,1)+1)*127.5).round() as u8, then HWC.
        u8 = ((np.clip(chw, -1.0, 1.0) + 1.0) * 127.5).round().astype("uint8")
        return np.transpose(u8, (1, 2, 0))

    for f in range(n):
        p = png_dir / f"{prefix}_frame{f}.png"
        Image.fromarray(to_u8_hwc(frame_chw(f))).save(str(p))
        print(f"wrote {p}", flush=True)

    # Contact sheet: near-square grid, black (-1.0) background.
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sheet = np.full((3, rows * h, cols * w), -1.0, dtype=np.float32)
    for f in range(n):
        gr, gc = divmod(f, cols)
        sheet[:, gr * h : gr * h + h, gc * w : gc * w + w] = frame_chw(f)
    p = png_dir / f"{prefix}_contact.png"
    Image.fromarray(to_u8_hwc(sheet)).save(str(p))
    print(f"wrote {p} ({cols}x{rows} grid {cols * w}x{rows * h})", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-noise", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--num-frames", required=True, type=int)
    p.add_argument("--steps", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--fps", type=int, default=24, help="real fps; bucket = 0 if fps==16 else 1")
    p.add_argument("--dtype", choices=list(DTYPES.keys()), default="bf16")
    p.add_argument(
        "--vae-dtype",
        choices=list(DTYPES.keys()),
        default="fp32",
        help="Wan VAE dtype (official example + f32 latents_mean/std default to fp32).",
    )
    p.add_argument(
        "--umt5-seq",
        type=int,
        default=0,
        help="Real (unpadded) umT5 token count. When >0, every umT5 tap dump is "
        "sliced to its first N rows so it aligns with the engine's real-token "
        "rows and the per-channel stats exclude masked padding.",
    )
    p.add_argument("--png-dir", type=Path, default=None)
    p.add_argument(
        "--png-prefix",
        default="py",
        help="Prefix for staged PNGs (py_frame{n}.png / py_contact.png), beside the rust ours_*.",
    )
    args = p.parse_args()

    if args.height % (VAE_SCALE * 2) or args.width % (VAE_SCALE * 2):
        raise SystemExit(f"height/width must be multiples of {VAE_SCALE * 2}")
    if (args.num_frames - 1) % TEMPORAL_SCALE != 0:
        raise SystemExit(f"num_frames-1 must be divisible by {TEMPORAL_SCALE}")
    args.out.mkdir(parents=True, exist_ok=True)

    f_lat = (args.num_frames - 1) // TEMPORAL_SCALE + 1
    h_lat = args.height // VAE_SCALE
    w_lat = args.width // VAE_SCALE
    n_expected = LATENT_CHANNELS * f_lat * h_lat * w_lat
    out_frames = TEMPORAL_SCALE * f_lat - 3

    raw = np.fromfile(str(args.initial_noise), dtype="<f4")
    if raw.size != n_expected:
        raise SystemExit(
            f"--initial-noise has {raw.size} f32 values; expected {n_expected} "
            f"for [1, {LATENT_CHANNELS}, {f_lat}, {h_lat}, {w_lat}]"
        )
    noise = raw.copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    vae_dtype = DTYPES[args.vae_dtype]
    fps_bucket = 0 if args.fps == 16 else 1
    print(
        f"device={device} dtype={dtype} vae_dtype={vae_dtype} fps={args.fps} "
        f"-> fps_bucket={fps_bucket} (rust FPS_BUCKET must equal this)",
        flush=True,
    )

    from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingPipeline

    t0 = time.time()
    # Load VAE in its own dtype (fp32) like the official example; transformer +
    # umT5 in the reference dtype (bf16, ~12 GB RAM ceiling for umT5-XXL).
    vae = AutoencoderKLWan.from_pretrained(
        REPO, subfolder="vae", torch_dtype=vae_dtype, local_files_only=True
    )
    pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        REPO, vae=vae, torch_dtype=dtype, local_files_only=True
    )
    print(f"loaded pipeline in {time.time() - t0:.1f}s (class={type(pipe).__name__})", flush=True)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # --- inject pinned noise as the starting latents (works around the
    # prepare_latents `latents is not None` early-return bug: run the random
    # path for the correct 4-tuple, then substitute our noise). ---
    orig_prepare = pipe.prepare_latents

    def hooked_prepare(*a, **k):
        a = list(a)
        if len(a) >= 9:
            a[8] = None  # force the random path -> correct (lat, nf, pvl, pvlf) tuple
        lat, nf, pvl, pvlf = orig_prepare(*a, **k)
        pinned = torch.from_numpy(noise).reshape(lat.shape).to(device=lat.device, dtype=lat.dtype)
        _summarize("starting latents (pinned noise injected)", pinned)
        return pinned, nf, pvl, pvlf

    pipe.prepare_latents = hooked_prepare  # type: ignore[assignment]

    # --- umt5_out: capture prompt_embeds [1, 512, 4096] (zero-padded to 512). ---
    orig_encode = pipe.encode_prompt

    def hooked_encode(*a, **k):
        pe, npe = orig_encode(*a, **k)
        _summarize("umt5_out (prompt_embeds, padded to 512)", pe)
        _dump(pe, args.out / "py_umt5_out.bin")
        return pe, npe

    pipe.encode_prompt = hooked_encode  # type: ignore[assignment]

    # --- per-umT5-layer + per-op taps: trace the encoder end to end so a clean
    # text-output scale (the ~0.365x bug) is localized to the exact op + block +
    # channel. Hooks every encoder submodule (forward hooks, captured on the
    # FIRST text-encoder forward only). The encoder runs the full 512-padded
    # sequence with an attention mask; each dump is sliced to the real `seq`
    # rows (--umt5-seq) so it aligns with the engine's real-token rows and the
    # per-channel stats exclude masked padding. All best-effort: a missing
    # submodule or shape surprise prints and skips. ---
    def _hook_umt5():
        te = getattr(pipe, "text_encoder", None)
        enc = getattr(te, "encoder", None) if te is not None else None
        if enc is None:
            print("  [py-tap] no text_encoder.encoder; skipping umT5 layer taps", flush=True)
            return
        fired = {"done": False}
        seq = args.umt5_seq

        def out_tensor(o):
            # Block/stack submodules return a tuple (hidden, ...); norms / linear
            # / embeds return a bare tensor.
            return o[0] if isinstance(o, (tuple, list)) else o

        def dump_seq(t: torch.Tensor, name: str) -> None:
            # Slice to the real-token rows (dim -2) so dumps match the engine's
            # `seq` rows and per-channel RMS ignores masked padding.
            if seq > 0 and t.dim() >= 2 and t.shape[-2] > seq:
                t = t[..., :seq, :]
            _summarize(f"umt5 {name}", t)
            _dump(t, args.out / f"py_umt5_{name}.bin")

        def dump_hook(name):
            def hook(_m, _inp, out):
                if fired["done"]:
                    return
                try:
                    dump_seq(out_tensor(out), name)
                except Exception as e:  # noqa: BLE001 - diagnostic best-effort
                    print(f"  [py-tap] umt5 {name} dump failed: {e}", flush=True)

            return hook

        def pre_hook(name):
            def hook(_m, inp):
                if fired["done"]:
                    return
                try:
                    dump_seq(out_tensor(inp), name)
                except Exception as e:  # noqa: BLE001
                    print(f"  [py-tap] umt5 {name} dump failed: {e}", flush=True)

            return hook

        def hook_path(root, path: str, name: str) -> int:
            # Resolve a dotted submodule path and register an output dump. Returns
            # 1 if hooked, 0 if the path is absent (printed + skipped).
            mod = root
            for part in path.split("."):
                mod = getattr(mod, part, None)
                if mod is None:
                    print(f"  [py-tap] no umt5 {path} ({name}); skipping", flush=True)
                    return 0
            mod.register_forward_hook(dump_hook(name))
            return 1

        n_hooked = 0
        embed = getattr(enc, "embed_tokens", None) or getattr(te, "shared", None)
        if embed is not None:
            embed.register_forward_hook(dump_hook("embeds"))
            n_hooked += 1

        # Per-block per-op trace. HF UMT5 block layout: layer.0 = self-attn
        # (layer_norm -> SelfAttention.{q,k,v,o}), layer.1 = gated FF (layer_norm
        # -> DenseReluDense.{wi_0,wi_1,wo}). These map to the engine's n1 / q / k
        # / v / proj / n2 / wi0 / wi1 / wo taps (sa/gu/after_attn have no single
        # HF module, covered by the block-output + neighbour ops).
        blocks = getattr(enc, "block", None)
        if blocks is not None:
            op_paths = [
                ("layer.0.layer_norm", "n1"),
                ("layer.0.SelfAttention.q", "q"),
                ("layer.0.SelfAttention.k", "k"),
                ("layer.0.SelfAttention.v", "v"),
                ("layer.0.SelfAttention.o", "o"),
                ("layer.1.layer_norm", "n2"),
                ("layer.1.DenseReluDense.wi_0", "wi0"),
                ("layer.1.DenseReluDense.wi_1", "wi1"),
                ("layer.1.DenseReluDense.wo", "wo"),
            ]
            for i, blk in enumerate(blocks):
                blk.register_forward_hook(dump_hook(f"block{i}_out"))
                n_hooked += 1
                for path, op in op_paths:
                    n_hooked += hook_path(blk, path, f"block{i}_{op}")

        fln = getattr(enc, "final_layer_norm", None)
        if fln is not None:
            # Input (== last block output, a cross-check) and output (== hidden).
            # The final norm fires LAST in the encoder forward, so its output
            # hook also closes the capture window: a second (negative-prompt)
            # encode then skips every hook and cannot overwrite the dumps.
            def hidden_hook(_m, _inp, out):
                dump_hook("hidden")(_m, _inp, out)
                fired["done"] = True

            fln.register_forward_pre_hook(pre_hook("final_in"))
            fln.register_forward_hook(hidden_hook)
            n_hooked += 1
        print(f"  [py-tap] hooked {n_hooked} umT5 submodules (seq slice={seq})", flush=True)

    # Real (unpadded) token count for clean tap slicing. Auto-compute from the
    # pipeline tokenizer when --umt5-seq is unset; it matches the engine's count
    # (same tokenizer.json + prompt), so both sides slice to the same rows.
    if args.umt5_seq <= 0 and getattr(pipe, "tokenizer", None) is not None:
        try:
            ids = pipe.tokenizer(args.prompt, add_special_tokens=True, return_tensors="pt").input_ids
            args.umt5_seq = int(ids.shape[-1])
            print(f"  [py-tap] auto umT5 seq={args.umt5_seq} (real tokens, unpadded)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [py-tap] umT5 seq auto-detect failed ({e}); dumping full padded rows", flush=True)

    _hook_umt5()

    # --- vae_rgb: capture vae.decode output [B, 3, F, H, W] in [-1, 1]. ---
    vae_out: dict[str, torch.Tensor] = {}
    orig_vae_decode = pipe.vae.decode

    def hooked_vae_decode(z, *a, **k):
        _summarize("vae.decode input (post-prescale latent)", z)
        result = orig_vae_decode(z, *a, **k)
        rgb = result.sample if hasattr(result, "sample") else result[0]
        _summarize("vae_rgb (vae.decode output, [-1,1])", rgb)
        vae_out["rgb"] = rgb.detach()
        _dump(rgb, args.out / "py_vae_rgb.bin")
        return result

    pipe.vae.decode = hooked_vae_decode  # type: ignore[assignment]

    # --- dit_out: capture the raw transformer output (flow velocity) per step,
    # i.e. the exact tensor handed to scheduler.step. This is the one tap that
    # cleanly splits a DiT-velocity slope from a scheduler-coefficient bug (the
    # parity suspects): our d.dit_out vs py_dit_out_step{i} is reconstruction-
    # free. Sync DF + no CFG => exactly one transformer forward per step. ---
    dit_step = {"i": 0}
    orig_transformer = pipe.transformer.forward

    def hooked_transformer(*a, **k):
        result = orig_transformer(*a, **k)
        if isinstance(result, (tuple, list)):
            out = result[0]
        else:
            out = getattr(result, "sample", result)
        i = dit_step["i"]
        _summarize(f"dit_out step{i} (transformer output, flow velocity)", out)
        _dump(out, args.out / f"py_dit_out_step{i}.bin")
        dit_step["i"] += 1
        return result

    pipe.transformer.forward = hooked_transformer  # type: ignore[assignment]

    # --- DEFENSIVE within-DiT taps (temb / timestep_proj / proj_out). These hook
    # diffusers-internal submodules whose exact signatures we are not pinning, so
    # each is wrapped: a mismatch skips that dump rather than crashing the whole
    # reference run. They only matter if the velocity tap above shows the slope is
    # inside the DiT (then they localize condition-embedder vs body vs proj_out).
    def _try_hook(mod_attr: str, dump_outputs):
        try:
            mod = pipe.transformer
            for part in mod_attr.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            print(f"  [py-tap] no submodule transformer.{mod_attr}; skipping", flush=True)
            return
        orig = mod.forward
        fired = {"done": False}

        def wrapped(*a, **k):
            r = orig(*a, **k)
            if not fired["done"]:
                fired["done"] = True
                try:
                    dump_outputs(r)
                except Exception as e:  # noqa: BLE001 - diagnostic best-effort
                    print(f"  [py-tap] transformer.{mod_attr} dump failed: {e}", flush=True)
            return r

        mod.forward = wrapped  # type: ignore[assignment]

    def _dump_cond_embedder(r):
        # diffusers Wan condition_embedder returns (temb, timestep_proj, enc_hidden,
        # enc_hidden_image) or similar; dump the first two if tuple-shaped.
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            _summarize("temb (condition_embedder[0])", r[0])
            _dump(r[0], args.out / "py_temb.bin")
            _summarize("timestep_proj (condition_embedder[1])", r[1])
            _dump(r[1], args.out / "py_timestep_proj.bin")

    def _dump_proj_out(r):
        t = r[0] if isinstance(r, (tuple, list)) else getattr(r, "sample", r)
        _summarize("proj_out (pre-unpatchify)", t)
        _dump(t, args.out / "py_proj_out.bin")

    _try_hook("condition_embedder", _dump_cond_embedder)
    _try_hook("proj_out", _dump_proj_out)

    # --- per-block taps: localize a DiT-body divergence to a specific block, at
    # EVERY step (the velocity slope first appears at step 1, so step-0-only block
    # dumps cannot localize it). Wrap every transformer block; each step dumps
    # every block's output residual as py_block{i}_out_step{s} (s = the shared
    # `dit_step` counter, which the transformer wrapper bumps AFTER each forward,
    # so it reads the current step inside the block call). On step 0 also capture
    # block 0's inputs (post-patch hidden_states == our patch_x; projected
    # encoder_hidden_states == our text_proj). All best-effort: any signature
    # mismatch prints and skips. Layout: diffusers blocks see [B=1, seq, inner];
    # _dump ravels to the same row-major [seq, inner] the rust per_block taps use. ---
    def _hook_blocks():
        try:
            blocks = pipe.transformer.blocks
        except AttributeError:
            print("  [py-tap] no transformer.blocks; skipping per-block taps", flush=True)
            return

        def make_wrapped(i, orig):
            def wrapped(*a, **k):
                s = dit_step["i"]
                if s == 0 and i == 0:
                    hs = a[0] if a else k.get("hidden_states")
                    ehs = a[1] if len(a) > 1 else k.get("encoder_hidden_states")
                    try:
                        if hs is not None:
                            _summarize("block-in hidden_states (post-patch == patch_x)", hs)
                            _dump(hs, args.out / "py_patch_in.bin")
                        if ehs is not None:
                            _summarize("block-in encoder_hidden_states (== text_proj)", ehs)
                            _dump(ehs, args.out / "py_text_in.bin")
                    except Exception as e:  # noqa: BLE001 - diagnostic best-effort
                        print(f"  [py-tap] block-0 input dump failed: {e}", flush=True)
                r = orig(*a, **k)
                out = r[0] if isinstance(r, (tuple, list)) else r
                try:
                    _dump(out, args.out / f"py_block{i}_out_step{s}.bin")
                except Exception as e:  # noqa: BLE001
                    print(f"  [py-tap] block{i} step{s} output dump failed: {e}", flush=True)
                return r

            return wrapped

        for i, blk in enumerate(blocks):
            blk.forward = make_wrapped(i, blk.forward)  # type: ignore[assignment]
        print(f"  [py-tap] hooked {len(blocks)} transformer blocks (per-step)", flush=True)

    _hook_blocks()

    # --- scheduler sub-updates: split a whole-step scheduler divergence into
    # convert / corrector / predictor so the order-2 predictor and corrector are
    # separately isolable (they first fire at step 1 and step 2 respectively under
    # lower_order_final). SkyReels DF deep-copies pipe.scheduler into per-frame
    # `sample_schedulers`, so we patch the CLASS (not the instance) -- every copy
    # then inherits the hook with correct `self` binding. Each per-frame call sees
    # this step's `self.step_index` (incremented only at the end of step), so we
    # bucket outputs by step_index and, after the run, stack the f_lat per-frame
    # [B,C,H,W] sub-outputs along the frame axis -> [B,C,F,H,W], matching the full
    # `py_step{i}_post` latent (and our Rust scheduler's flattened [z,f,h,w]). Dump:
    #   py_sched_conv_step{i}  convert_model_output (x0 prediction m_conv)
    #   py_sched_corr_step{i}  multistep_uni_c_bh_update (absent on step 0)
    #   py_sched_pred_step{i}  multistep_uni_p_bh_update (== py_step{i}_post)
    sched_buf: dict[str, dict[int, list]] = {"conv": {}, "corr": {}, "pred": {}}
    sched_orders: dict[int, int] = {}

    def _hook_scheduler():
        cls = type(pipe.scheduler)
        names = {
            "convert_model_output": "conv",
            "multistep_uni_c_bh_update": "corr",
            "multistep_uni_p_bh_update": "pred",
        }
        originals = {}
        for name, tag in names.items():
            if not hasattr(cls, name):
                print(f"  [py-tap] {cls.__name__} has no {name}; skipping", flush=True)
                continue
            orig = getattr(cls, name)
            originals[name] = orig

            def make(orig, tag):
                def wrapped(self, *a, **k):
                    r = orig(self, *a, **k)
                    try:
                        idx = int(self.step_index)
                        sched_buf[tag].setdefault(idx, []).append(r.detach().clone())
                        if tag == "pred":
                            sched_orders[idx] = int(self.this_order)
                    except Exception as e:  # noqa: BLE001 - diagnostic best-effort
                        print(f"  [py-tap] scheduler {tag} bucket failed: {e}", flush=True)
                    return r

                return wrapped

            setattr(cls, name, make(orig, tag))
        print(f"  [py-tap] hooked {cls.__name__} convert/corrector/predictor", flush=True)
        return cls, originals

    sched_cls, sched_orig = _hook_scheduler()

    # --- step{i}_post + pre_vae_latent via callback (latents after each
    # scheduler.step; the final one is the pre-prescale latent the Rust side
    # returns as our_pre_vae). ---
    step_state = {"last": None}

    def on_step_end(_pipe, i, _t, cb_kwargs):
        lat = cb_kwargs["latents"]
        _summarize(f"step{i}_post (latent after scheduler.step)", lat)
        _dump(lat, args.out / f"py_step{i}_post.bin")
        step_state["last"] = lat.detach()
        return cb_kwargs

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.time()
    out = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=1.0,  # no CFG: one DiT forward per step (matches engine)
        generator=gen,
        fps=args.fps,
        ar_step=0,  # synchronous DF
        causal_block_size=None,  # -> num_frame_per_block stays 1 -> full attention
        overlap_history=None,  # short video, single iteration
        output_type="np",
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    print(f"denoised + decoded in {time.time() - t1:.1f}s", flush=True)

    # Restore the patched class methods, then reassemble + dump the per-step
    # scheduler sub-outputs. Each bucket[i] is the f_lat per-frame [B,C,H,W]
    # outputs in frame order; stack along dim=2 -> [B,C,F,H,W] (ravels to the same
    # row-major [z,f,h,w] the Rust scheduler diag uses).
    for name, orig in sched_orig.items():
        setattr(sched_cls, name, orig)
    for tag, prefix in (("conv", "py_sched_conv"), ("corr", "py_sched_corr"), ("pred", "py_sched_pred")):
        for i, frames in sorted(sched_buf[tag].items()):
            stacked = torch.stack(frames, dim=2)  # [B,C,F,H,W]
            _dump(stacked, args.out / f"{prefix}_step{i}.bin")
    if sched_orders:
        print(f"  [py-tap] scheduler orders: {dict(sorted(sched_orders.items()))}", flush=True)

    # pre_vae_latent == the final post-step latent (pre mean/std prescale), the
    # exact tensor the Rust side returns as our_pre_vae before decode.
    if step_state["last"] is not None:
        _summarize("pre_vae_latent (final denoised latent, pre-prescale)", step_state["last"])
        _dump(step_state["last"], args.out / "py_pre_vae_latent.bin")

    # PNGs from the raw vae output (batch-squeezed [3, F, H, W]); same path the
    # Rust ours_* frames take, so py_*/ours_* are visually comparable.
    if args.png_dir is not None and "rgb" in vae_out:
        rgb = vae_out["rgb"].to(torch.float32).cpu().numpy()
        rgb = rgb[0] if rgb.ndim == 5 else rgb  # [B,3,F,H,W] -> [3,F,H,W]
        if rgb.shape == (3, out_frames, args.height, args.width):
            _stage_frames(rgb, out_frames, args.height, args.width, args.png_dir, args.png_prefix)
        else:
            print(
                f"  [warn] vae rgb shape {rgb.shape} != "
                f"(3,{out_frames},{args.height},{args.width}); skipping PNG stage",
                flush=True,
            )

    for name in ("py_umt5_out.bin", "py_pre_vae_latent.bin", "py_vae_rgb.bin"):
        fp = args.out / name
        if fp.exists():
            print(f"wrote {fp} ({fp.stat().st_size} bytes)", flush=True)
    print("[py-video-e2e] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
