"""End-to-end FastWan2.2-TI2V-5B reference: prompt + pinned noise -> RGB video.

The FOREVER parity reference for the Wan video pipeline. Mirrors the engine's
`WanModel::denoise_with` + `decode_latent_to_video` exactly so that, given the
same prompt / dims / initial noise / per-step renoise, every captured stage is
byte-comparable. Unlike Z-Image (whose scheduler is deterministic on its own),
DMD re-noises between steps; we make it byte-parity-friendly by having the Rust
side DUMP the exact initial latent + each per-step renoise tensor and feeding
them in here, so neither side draws from its own RNG.

What this dumps (little-endian f32, full tensors; CTHW latent = [z, f, h, w]):

  py_dit_out_step{i}.bin   raw transformer output (flow velocity) at step i,
                           == engine `WanStepDiag.velocity`.
  py_step{i}_post.bin      latent after the DMD step (x0 renoised to the next
                           sigma; x0 itself on the last step), == engine
                           `WanStepDiag.post`. The last step == py_pre_vae.
  py_pre_vae_latent.bin    final pre-VAE latent (raw, before un-normalize), ==
                           engine `denoise_with` return.
  py_vae_rgb.bin           VAE-decoded video, CTHW f32 [3, frames, H, W] in
                           [-1, 1], == engine `decode_latent_to_video`.

DiT-internal taps (for bisecting a DiT divergence against the engine's
`diag_step_at`; dumped for ONE DMD step, THINFER_WAN_DIAG_STEP (default 0 ==
t=1000), which is the step the
engine diag reproduces). Each maps 1:1 to a `WanStep0Diag` field:

  py_in_umt5_hidden.bin    umT5 last_hidden_state, real tokens [seq, 4096].
  py_in_patch_x.bin        post-patchify tokens [n_tok, inner].
  py_in_temb.bin           time embedding [inner].
  py_in_timestep_proj.bin  projected timestep [6, inner] (row-major == engine).
  py_in_text_proj.bin      text states after condition_embedder [512, inner].
  py_in_block{ii}.bin      residual stream after block ii [n_tok, inner], ii=0..N.
  py_in_final_norm.bin     post norm_out modulation, == proj_out input [n_tok, inner].
  py_in_proj_out.bin       proj_out output [n_tok, out_ch*p_t*p_h*p_w].

DMD math matches `wan/scheduler.rs`: sigma = t / num_train_timesteps,
x0 = x_t - sigma*v, x_{i+1} = (1 - sigma_next)*x0 + sigma_next*noise. (FastVideo
snaps sigma against a shift=8 FlowMatch grid, which returns t/1000 to within
~0.03%: t=757 -> 0.7568 vs 0.757, t=522 -> 0.5217 vs 0.522. That snap is a
bounded delta the parity test confirms is absorbed; the engine uses plain
t/1000 by decision, so the reference matches the engine here.)

Precision: bf16 DiT + umT5 (matches the engine's bf16 weights / f16 acts path),
fp32 VAE (the engine decodes the VAE in fp32). DiT step math is done in fp32 on
the bf16 transformer output, mirroring the engine's f32 readback of velocity.

Usage:

    uv run python -m thinfer_pytorch_ref.wan.gen_fastwan_video_e2e_ref \\
        --initial-noise <noise.bin> \\
        --renoise <renoise_step0.bin> [--renoise <renoise_step1.bin> ...] \\
        --transformer-shard <dit.safetensors> \\
        --prompt "..." --height 32 --width 32 --frames 5 --seed 42 \\
        --out <tmpdir>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Fixed by the model; must match `wan/pipeline.rs` + `tests/wan/video_e2e.rs`.
Z_DIM = 48
VAE_SCALE = 16
TEMPORAL_SCALE = 4
DENOISING_STEPS = [1000.0, 757.0, 522.0]
NUM_TRAIN_TIMESTEPS = 1000.0
TEXT_SEQ = 512
TEXT_DIM = 4096


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _summarize(label: str, t: torch.Tensor) -> None:
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    print(
        f"  [PY-DUMP] {label}: len={a.size} min={a.min():.5e} max={a.max():.5e} "
        f"max_abs={abs(a).max():.5e} mean={a.mean():.5e}",
        flush=True,
    )


def _load(path: Path, n: int, name: str) -> np.ndarray:
    raw = np.fromfile(str(path), dtype="<f4")
    if raw.size != n:
        raise SystemExit(f"--{name} {path} has {raw.size} f32 values; expected {n}")
    return raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-noise", required=True, type=Path)
    p.add_argument(
        "--renoise",
        type=Path,
        action="append",
        default=[],
        help="Per-step renoise tensor (one per non-final DMD step), in order.",
    )
    p.add_argument(
        "--transformer-shard",
        required=True,
        type=Path,
        action="append",
        help="Safetensors file(s) holding the DiT state dict. Repeat for shards.",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--frames", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--dtype", choices=list(DTYPES.keys()), default="bf16")
    args = p.parse_args()

    div = VAE_SCALE * 2  # VAE 16x spatial * DiT patch 2 -> 32px granularity.
    if args.height % div or args.width % div:
        raise SystemExit(f"height/width must be multiples of {div}; got {args.height}x{args.width}")
    if args.frames % TEMPORAL_SCALE != 1:
        raise SystemExit(f"frames must be 4k+1; got {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    f_lat = (args.frames - 1) // TEMPORAL_SCALE + 1
    h_lat = args.height // VAE_SCALE
    w_lat = args.width // VAE_SCALE
    n_lat = Z_DIM * f_lat * h_lat * w_lat
    n_steps = len(DENOISING_STEPS)
    if len(args.renoise) != n_steps - 1:
        raise SystemExit(f"expected {n_steps - 1} --renoise tensors, got {len(args.renoise)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    print(
        f"device={device} dtype={dtype} grid f_lat={f_lat} h_lat={h_lat} w_lat={w_lat} "
        f"n_lat={n_lat}",
        flush=True,
    )

    shape = (1, Z_DIM, f_lat, h_lat, w_lat)
    noise = torch.from_numpy(_load(args.initial_noise, n_lat, "initial-noise").copy()).reshape(shape)
    renoise = [
        torch.from_numpy(_load(r, n_lat, "renoise").copy()).reshape(shape) for r in args.renoise
    ]

    from diffusers import WanPipeline
    from safetensors.torch import load_file as load_safetensors

    t0 = time.time()
    pipe = WanPipeline.from_pretrained(REPO, torch_dtype=dtype, local_files_only=True)
    print(f"loaded pipeline in {time.time() - t0:.1f}s (class={type(pipe).__name__})", flush=True)

    # Override transformer weights from the exact shard the engine loads; refuse
    # a partial load so any divergence is attributable to engine math, not
    # weight provenance (same discipline as the Z-Image e2e ref).
    merged: dict[str, torch.Tensor] = {}
    for shard in args.transformer_shard:
        for k, v in load_safetensors(str(shard)).items():
            if k in merged:
                raise SystemExit(f"duplicate key {k!r} across shards")
            merged[k] = v
    merged = {k: v.to(dtype) for k, v in merged.items()}
    missing, unexpected = pipe.transformer.load_state_dict(merged, strict=False)
    if missing or unexpected:
        print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        raise SystemExit("transformer override key mismatch; refusing partial weights")
    print(f"transformer override: {len(merged)} keys", flush=True)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.transformer.eval()
    pipe.text_encoder.eval()

    # --- text encode (inlined _get_t5_prompt_embeds: tokenize -> umT5 ->
    # per-sample real-token slice -> zero-pad to TEXT_SEQ). Matches the engine's
    # umT5 forward + `pad_text`. No cross-attn mask reaches the transformer. ---
    text_inputs = pipe.tokenizer(
        args.prompt,
        padding="max_length",
        max_length=TEXT_SEQ,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        embeds = pipe.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
    embeds = embeds.to(dtype=dtype)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    # umT5 output, real tokens == engine `WanStep0Diag.umt5_hidden`.
    _dump(embeds[0], args.out / "py_in_umt5_hidden.bin")
    _summarize("umt5_hidden (real tokens)", embeds[0])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(TEXT_SEQ - u.size(0), u.size(1))]) for u in embeds], dim=0
    ).to(device)
    _summarize("prompt_embeds (padded text context)", prompt_embeds)
    print(f"text tokens={int(seq_lens[0])} (padded to {TEXT_SEQ})", flush=True)

    # --- DiT-internal taps. Forward hooks on the transformer modules that map
    # 1:1 to the engine `diag_step_at` stage readbacks. Gated on a flag so only
    # ONE DMD step is captured (later steps would overwrite the same files). The
    # captured step is THINFER_WAN_DIAG_STEP (default 0 == t=1000, the step the
    # engine `diag_step0` reproduces); point it at the first divergent step (e.g.
    # 1 == t=757) to bisect a timestep-specific divergence. The engine feeds its
    # forward the matching input (py_step{N-1}_post) so both share an input. ---
    diag_step = int(os.environ.get("THINFER_WAN_DIAG_STEP", "0"))
    tr = pipe.transformer
    cap = {"on": False}

    def _hook_patch(_m, _inp, out):
        if cap["on"]:
            # conv [1, inner, ppf, pph, ppw] -> tokens [1, n_tok, inner].
            _dump(out.flatten(2).transpose(1, 2), args.out / "py_in_patch_x.bin")

    def _hook_cond(_m, _inp, out):
        if cap["on"]:
            temb, timestep_proj, enc, _img = out
            _dump(temb, args.out / "py_in_temb.bin")
            _dump(timestep_proj, args.out / "py_in_timestep_proj.bin")
            _dump(enc, args.out / "py_in_text_proj.bin")

    def _hook_block(idx):
        def hook(_m, _inp, out):
            if cap["on"]:
                _dump(out, args.out / f"py_in_block{idx:02d}.bin")
        return hook

    def _hook_proj(_m, inp, out):
        if cap["on"]:
            _dump(inp[0], args.out / "py_in_final_norm.bin")
            _dump(out, args.out / "py_in_proj_out.bin")

    handles = [
        tr.patch_embedding.register_forward_hook(_hook_patch),
        tr.condition_embedder.register_forward_hook(_hook_cond),
        tr.proj_out.register_forward_hook(_hook_proj),
    ]
    handles += [b.register_forward_hook(_hook_block(i)) for i, b in enumerate(tr.blocks)]

    # --- block-0 internal taps: an instrumented copy of WanTransformerBlock.forward
    # (uniform-t / temb.ndim==3 path) that dumps the module-boundary stages 1:1 with
    # the engine `diag_step0` block0 readbacks (norm1/after_self/norm2/after_cross/
    # norm3/ffn_gelu/ffn_down). Used only when cap["on"] (step 0); otherwise the
    # original forward runs. The block's own forward hook still dumps py_in_block00. ---
    b0 = tr.blocks[0]
    orig_b0_forward = b0.forward

    def _hook_b0_gelu(_m, _inp, out):
        if cap["on"]:
            _dump(out, args.out / "py_b0_ffn_gelu.bin")

    handles.append(b0.ffn.net[0].register_forward_hook(_hook_b0_gelu))

    # block-0 self-attn (attn1) internal taps, 1:1 with the engine `block0_stages`
    # self_q/k/v/sa. to_q/k/v outputs are post-projection+bias, pre-qk-norm/pre-rope
    # (engine taps `self_*` before norm_q); to_out[0]'s input is the flattened
    # post-sdpa context, pre-output-projection (engine tap `self_sa`).
    def _hook_b0_self(nm):
        def hook(_m, _inp, out):
            if cap["on"]:
                _dump(out, args.out / f"py_b0_self_{nm}.bin")
        return hook

    def _hook_b0_self_sa(_m, inp):
        if cap["on"]:
            _dump(inp[0], args.out / "py_b0_self_sa.bin")

    handles.append(b0.attn1.to_q.register_forward_hook(_hook_b0_self("q")))
    handles.append(b0.attn1.to_k.register_forward_hook(_hook_b0_self("k")))
    handles.append(b0.attn1.to_v.register_forward_hook(_hook_b0_self("v")))
    handles.append(b0.attn1.to_out[0].register_forward_pre_hook(_hook_b0_self_sa))

    def _b0_forward(hidden_states, encoder_hidden_states, temb, rotary_emb):
        if not cap["on"]:
            return orig_b0_forward(hidden_states, encoder_hidden_states, temb, rotary_emb)
        # temb == timestep_proj [B, 6, inner]; modulation broadcasts over tokens.
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            b0.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        for nm, vv in [
            ("shift", shift_msa), ("scale", scale_msa), ("gate", gate_msa),
            ("cshift", c_shift_msa), ("cscale", c_scale_msa), ("cgate", c_gate_msa),
        ]:
            _dump(vv, args.out / f"py_b0_mod_{nm}.bin")
        premod = b0.norm1(hidden_states.float())
        _dump(premod, args.out / "py_b0_norm1_premod.bin")
        norm_hs = (premod * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        _dump(norm_hs, args.out / "py_b0_norm1.bin")
        attn_output = b0.attn1(norm_hs, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        _dump(hidden_states, args.out / "py_b0_after_self.bin")
        norm_hs = b0.norm2(hidden_states.float()).type_as(hidden_states)
        _dump(norm_hs, args.out / "py_b0_norm2.bin")
        attn_output = b0.attn2(norm_hs, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output
        _dump(hidden_states, args.out / "py_b0_after_cross.bin")
        norm_hs = (b0.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        _dump(norm_hs, args.out / "py_b0_norm3.bin")
        ff_output = b0.ffn(norm_hs)
        _dump(ff_output, args.out / "py_b0_ffn_down.bin")
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states

    b0.forward = _b0_forward

    # --- DMD few-step loop (matches wan/scheduler.rs) ---
    latents = noise.to(dtype=dtype, device=device)
    sigmas = [t / NUM_TRAIN_TIMESTEPS for t in DENOISING_STEPS]
    for i, t in enumerate(DENOISING_STEPS):
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
        cap["on"] = i == diag_step
        with torch.no_grad():
            # diffusers WanTransformer3DModel.forward(hidden_states, timestep,
            # encoder_hidden_states). 1D timestep -> scalar (uniform-t) path,
            # which is what the engine drives for T2V.
            velocity = pipe.transformer(latents, t_tensor, prompt_embeds, return_dict=False)[0]
        cap["on"] = False
        _summarize(f"dit_out_step{i} (velocity, t={t:.0f})", velocity)
        _dump(velocity, args.out / f"py_dit_out_step{i}.bin")

        x0 = latents.float() - sigmas[i] * velocity.float()
        if i + 1 < n_steps:
            sigma_next = sigmas[i + 1]
            n = renoise[i].float().to(device)
            latents = ((1.0 - sigma_next) * x0 + sigma_next * n).to(dtype=dtype)
        else:
            latents = x0.to(dtype=dtype)
        _summarize(f"step{i}_post", latents)
        _dump(latents, args.out / f"py_step{i}_post.bin")

    for h in handles:
        h.remove()

    pre_vae = latents
    _dump(pre_vae, args.out / "py_pre_vae_latent.bin")
    _summarize("pre_vae_latent (raw, pre-unnormalize)", pre_vae)

    # --- VAE decode (fp32). Un-normalize first: z_raw * std + mean, per the
    # diffusers WanPipeline (latents / (1/std) + mean). Then decode -> CTHW RGB
    # clamped to [-1, 1]. ---
    vae = pipe.vae.to(dtype=torch.float32, device=device)
    vae.eval()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, Z_DIM, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(1, Z_DIM, 1, 1, 1)
    z = pre_vae.float().to(device) * std + mean
    with torch.no_grad():
        video = vae.decode(z, return_dict=False)[0]  # [1, 3, frames, H, W]
    rgb = video[0]  # drop batch -> [3, frames, H, W], == engine layout
    _summarize("vae_rgb (CTHW [-1, 1])", rgb)
    _dump(rgb, args.out / "py_vae_rgb.bin")

    for name in ("py_pre_vae_latent.bin", "py_vae_rgb.bin"):
        pth = args.out / name
        print(f"wrote {pth} ({pth.stat().st_size} bytes)", flush=True)
    print("[py-fastwan-e2e] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
