"""End-to-end LongLive-2.0-5B AR (causal) reference: prompt + pinned noise -> RGB.

The parity reference for the LongLive autoregressive video path. Unlike the
FastWan ref (which drives diffusers' full-attention WanPipeline), the AR
backbone is upstream-specific, so this drives the AUTHORITATIVE upstream
`CausalWanModel` (`third-party/LongLive/wan_5b/modules/causal_model.py`) on the
real `model_bf16.pt`, and reimplements ONLY the small single-prompt T2V chunk
loop from `pipeline/causal_diffusion_inference.py::_inference_inner` (no CFG, no
i2v, no scene cuts -> no pinning, rope_offset 0). The cache rolling / windowing /
absolute-RoPE all live inside `CausalWanModel.forward`, so they are upstream
truth, not reimplemented here.

The shared base (umT5 text encoder + Wan2.2 VAE) is loaded from the diffusers
FastWan repo exactly as the FastWan ref does, because the engine reuses that
same base for LongLive (the "same weights" parity is honest: the DiT is the
real `.pt`, the base is the shared bundle).

Runs on CPU in bf16 (the venv is torch-CPU; fp32 would blow host RAM -- the .pt
is 10GB). Triton RoPE/adaLN kernels are disabled via env so the eager CPU path
runs; flash-attn is absent so upstream `attention()` falls back to SDPA. The
flex_attention path is training-only (kv_cache=None) and never reached here.

Layout note: the engine latent is CTHW `[C, f_lat, h, w]`; the upstream pipeline
is frame-major `[B, F, C, H, W]`. The pinned noise is shared CTHW; we reshape +
permute it into frame-major for the loop, and dump every latent back in CTHW so
the bytes line up with the engine.

What this dumps (little-endian f32):
  py_umt5_hidden.bin     umT5 last_hidden_state, real tokens [seq, 4096].
  py_chunk{c}_post.bin   denoised latent for chunk c after the 4 UniPC steps,
                         CTHW [C, chunk_frames, h, w] (localizes a per-chunk
                         divergence: chunk 0 is FastWan-like; chunk>=1 exercises
                         the committed-window prefix attention + absolute RoPE).
  py_pre_vae_latent.bin  full assembled pre-VAE latent, CTHW [C, f_lat, h, w] ==
                         engine `denoise_ar` return.
  py_vae_rgb.bin         VAE-decoded video, CTHW f32 [3, frames, H, W] in [-1, 1].

Usage:
    uv run python -m thinfer_pytorch_ref.wan.gen_longlive_video_e2e_ref \\
        --initial-noise <noise.bin> --generator-ckpt <model_bf16.pt> \\
        --prompt "..." --height 128 --width 128 --frames 61 --seed 42 \\
        --out <tmpdir>
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Disable upstream's Triton/CUDA fast paths so the eager CPU path runs.
os.environ.setdefault("LLV2_TRITON_ROPE", "0")
os.environ.setdefault("LLV2_TRITON_ADALN", "0")
os.environ.setdefault("LLV2_FREQS_I_CACHE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
LONGLIVE_CLONE = Path(
    os.environ.get("THINFER_LONGLIVE_CLONE", r"C:\work\personal\thinfer\third-party\LongLive")
)

# Fixed by the model; must match the engine + the Rust test.
Z_DIM = 48
VAE_SCALE = 16
TEMPORAL_SCALE = 4
TEXT_SEQ = 512
TEXT_DIM = 4096
CHUNK_FRAMES = 8  # num_frame_per_block
DIM = 3072
NUM_HEADS = 24
FFN_DIM = 14336
NUM_LAYERS = 30
HEAD_DIM = DIM // NUM_HEADS
# Release inference.yaml (configs/inference.yaml).
LOCAL_ATTN_SIZE = 32
SINK_SIZE = 8
TIMESTEP_SHIFT = 5.0
SAMPLING_STEPS = 4
NUM_TRAIN_TIMESTEPS = 1000


def _dump(t: torch.Tensor, path: Path) -> None:
    t.detach().to(torch.float32).cpu().numpy().astype("<f4").tofile(str(path))


def _summarize(label: str, t: torch.Tensor) -> None:
    a = t.detach().to(torch.float32).cpu().numpy().ravel()
    print(
        f"  [PY-DUMP] {label}: len={a.size} min={a.min():.5e} max={a.max():.5e} "
        f"max_abs={abs(a).max():.5e} mean={a.mean():.5e}",
        flush=True,
    )


def _base_repo_dir() -> str:
    """Local snapshot dir of the shared FastWan base (umT5 + VAE + tokenizer).
    Loading from a directory path avoids any HF hub lookup. THINFER_FASTWAN_DIR
    overrides (the Rust test sets it from the resolved cache path)."""
    env = os.environ.get("THINFER_FASTWAN_DIR")
    if env:
        return env
    base = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / f"models--{REPO.replace('/', '--')}" / "snapshots"
    )
    snaps = sorted(p for p in base.glob("*") if p.is_dir())
    if not snaps:
        raise SystemExit(f"FastWan base snapshot not found under {base}")
    return str(snaps[-1])


def _load(path: Path, n: int, name: str) -> np.ndarray:
    raw = np.fromfile(str(path), dtype="<f4")
    if raw.size != n:
        raise SystemExit(f"--{name} {path} has {raw.size} f32 values; expected {n}")
    return raw


def _setup_upstream_imports():
    """Import upstream CausalWanModel + UniPC scheduler from the LongLive clone."""
    sys.path.insert(0, str(LONGLIVE_CLONE))
    # This transformers version dropped x_clip_loss, which causal_model imports
    # at module top but never uses on the inference path. Shim it (the clone is
    # read-only).
    import transformers.models.x_clip.modeling_x_clip as _xclip

    if not hasattr(_xclip, "x_clip_loss"):
        _xclip.x_clip_loss = lambda *a, **k: None

    import wan_5b.modules.causal_model as cm
    from wan_5b.modules.model import rope_params
    from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # The cross-attn calls `flash_attention` DIRECTLY (CUDA-only assert), unlike
    # self-attn which goes through `attention()` (SDPA fallback when flash-attn is
    # absent). Replace it with the same SDPA path so the CPU run works. No
    # cross-attn mask (k_lens ignored) -- matches the engine + FastWan ref, which
    # attend the full zero-padded 512 text context with no mask.
    def _sdpa_flash_attention(
        q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None,
        q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
        dtype=torch.bfloat16, version=None,
    ):
        out_dtype = q.dtype
        if q_scale is not None:
            q = q * q_scale
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=None, is_causal=causal, dropout_p=0.0,
            scale=softmax_scale,
        )
        return o.transpose(1, 2).to(out_dtype)

    cm.flash_attention = _sdpa_flash_attention

    return cm, rope_params, FlowUniPCMultistepScheduler


def _build_causal_model(cm, rope_params, ckpt_path: Path):
    """Meta-instantiate CausalWanModel + load the .pt generator weights in-place
    (assign=True avoids a 2x10GB transient). Weights are bf16; freqs (a plain
    non-buffer attribute) is meta after construction, so recompute it on CPU."""
    cm.CausalWanModel.init_weights = lambda self: None  # weights come from the .pt
    with torch.device("meta"):
        model = cm.CausalWanModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=TEXT_SEQ,
            in_dim=Z_DIM, dim=DIM, ffn_dim=FFN_DIM, freq_dim=256, text_dim=TEXT_DIM,
            out_dim=Z_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            local_attn_size=LOCAL_ATTN_SIZE, sink_size=SINK_SIZE,
            num_frame_per_block=CHUNK_FRAMES, qk_norm=True, cross_attn_norm=True,
            eps=1e-6,
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=True)
    gen = ckpt["generator"]
    state = {k[len("model."):]: v for k, v in gen.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, assign=True, strict=True)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval()
    # Recompute freqs on CPU (meta after construction; not a parameter/buffer).
    d = HEAD_DIM
    model.freqs = torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1)
    return model


def _set_inference_attrs(model, frame_seq_length: int):
    """Mirror CausalDiffusionInferencePipeline.inference()'s runtime overrides for
    the release single-prompt T2V case (multi_shot_sink=true -> global_sink=sink)."""
    global_sink = SINK_SIZE  # multi_shot_sink: true => global_sink_size = sink_size
    target_max = LOCAL_ATTN_SIZE * frame_seq_length
    for module in model.modules():
        if hasattr(module, "max_attention_size"):
            module.max_attention_size = target_max
        if hasattr(module, "sink_size"):
            module.sink_size = SINK_SIZE
        if hasattr(module, "global_sink_size"):
            module.global_sink_size = global_sink
    if hasattr(model, "max_attention_size"):
        model.max_attention_size = target_max
    model.global_sink_size = global_sink
    model.local_attn_size = LOCAL_ATTN_SIZE
    model.t_scale = 1.0
    model.rope_method = "linear"
    model.original_seq_len = None
    model.use_relative_rope = False
    model.rope_temporal_offset = 0.0


def _init_kv_cache(frame_seq_length: int, dtype, device):
    kv_cache_size = LOCAL_ATTN_SIZE * frame_seq_length
    block_token_size = CHUNK_FRAMES * frame_seq_length
    max_blocks = kv_cache_size // block_token_size
    kv_cache, crossattn_cache = [], []
    for _ in range(NUM_LAYERS):
        kv_cache.append({
            "k": torch.zeros([1, kv_cache_size, NUM_HEADS, HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([1, kv_cache_size, NUM_HEADS, HEAD_DIM], dtype=dtype, device=device),
            "quantized": False,
            "block_token_size": block_token_size,
            "max_blocks": max_blocks,
            "num_heads": NUM_HEADS,
            "num_filled_blocks": 0,
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
            "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
        })
        crossattn_cache.append({
            "k": torch.zeros([1, TEXT_SEQ, NUM_HEADS, HEAD_DIM], dtype=dtype, device=device),
            "v": torch.zeros([1, TEXT_SEQ, NUM_HEADS, HEAD_DIM], dtype=dtype, device=device),
            "is_init": False,
        })
    return kv_cache, crossattn_cache


def _encode_text(prompt: str, dtype, device, out_dir: Path) -> torch.Tensor:
    from transformers import AutoTokenizer, UMT5EncoderModel

    repo_dir = _base_repo_dir()
    tok = AutoTokenizer.from_pretrained(repo_dir, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(
        repo_dir, subfolder="text_encoder", torch_dtype=dtype
    ).eval()
    ti = tok(
        prompt, padding="max_length", max_length=TEXT_SEQ, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    ids, mask = ti.input_ids, ti.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        embeds = te(ids.to(device), mask.to(device)).last_hidden_state.to(dtype)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    _dump(embeds[0], out_dir / "py_umt5_hidden.bin")
    _summarize("umt5_hidden (real tokens)", embeds[0])
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(TEXT_SEQ - u.size(0), u.size(1))]) for u in embeds], dim=0
    ).to(device)
    print(f"text tokens={int(seq_lens[0])} (padded to {TEXT_SEQ})", flush=True)
    del te, tok
    gc.collect()
    return prompt_embeds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--initial-noise", required=True, type=Path)
    p.add_argument("--generator-ckpt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prompt", required=True)
    p.add_argument("--height", required=True, type=int)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--frames", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()

    div = VAE_SCALE * 2
    if args.height % div or args.width % div:
        raise SystemExit(f"height/width must be multiples of {div}; got {args.height}x{args.width}")
    if args.frames % TEMPORAL_SCALE != 1:
        raise SystemExit(f"frames must be 4k+1; got {args.frames}")
    args.out.mkdir(parents=True, exist_ok=True)

    f_lat = (args.frames - 1) // TEMPORAL_SCALE + 1
    h_lat = args.height // VAE_SCALE
    w_lat = args.width // VAE_SCALE
    n_lat = Z_DIM * f_lat * h_lat * w_lat
    frame_seq_length = (h_lat // 2) * (w_lat // 2)
    if f_lat % CHUNK_FRAMES != 0:
        raise SystemExit(f"f_lat ({f_lat}) must be a multiple of {CHUNK_FRAMES}")
    num_chunks = f_lat // CHUNK_FRAMES

    device = "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(
        f"device={device} dtype={dtype} f_lat={f_lat} h_lat={h_lat} w_lat={w_lat} "
        f"n_lat={n_lat} frame_seq_length={frame_seq_length} num_chunks={num_chunks}",
        flush=True,
    )

    cm, rope_params, FlowUniPCMultistepScheduler = _setup_upstream_imports()

    # --- pinned noise: shared CTHW [C, f_lat, h, w] -> frame-major [B, F, C, h, w] ---
    noise_flat = _load(args.initial_noise, n_lat, "initial-noise").copy()
    noise_cfhw = torch.from_numpy(noise_flat).reshape(1, Z_DIM, f_lat, h_lat, w_lat)
    noise_fcHW = noise_cfhw.permute(0, 2, 1, 3, 4).contiguous().to(dtype)  # [B, F, C, h, w]

    # --- 1. text encode (umT5, freed before the DiT loads) ---
    t0 = time.time()
    prompt_embeds = _encode_text(args.prompt, dtype, device, args.out)
    print(f"text encode done in {time.time() - t0:.1f}s", flush=True)

    # --- 2. DiT (CausalWanModel) ---
    t0 = time.time()
    model = _build_causal_model(cm, rope_params, args.generator_ckpt)
    _set_inference_attrs(model, frame_seq_length)
    print(f"DiT loaded in {time.time() - t0:.1f}s", flush=True)

    kv_cache, crossattn_cache = _init_kv_cache(frame_seq_length, dtype, device)
    output = torch.zeros([1, f_lat, Z_DIM, h_lat, w_lat], dtype=dtype, device=device)

    # Per-block residual taps (chunk0/step0 only) for localization. The block
    # output is token-space [B, n_tok, inner]; dump [n_tok, inner] to match the
    # engine residual stream. The kv-cache path returns (x, cache_info) tuples.
    cap = {"on": False}

    def _mk_block_hook(i):
        def hook(_m, _inp, out):
            if cap["on"]:
                x = out[0] if isinstance(out, tuple) else out
                _dump(x[0], args.out / f"py_c0s0_block{i}.bin")
        return hook

    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(_mk_block_hook(i))

    def run_model(latents_bfchw, timestep_bf, current_start):
        # Mirror WanDiffusionWrapper.forward (causal: input_timestep == full [B,F]).
        x = latents_bfchw.permute(0, 2, 1, 3, 4)  # [B, C, F, h, w]
        out = model(
            x, t=timestep_bf, context=prompt_embeds, seq_len=28160,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start=current_start, cache_start=current_start,
        )
        return out.permute(0, 2, 1, 3, 4)  # [B, F, C, h, w]

    t0 = time.time()
    for chunk in range(num_chunks):
        f0 = chunk * CHUNK_FRAMES
        current_start = chunk * CHUNK_FRAMES * frame_seq_length
        # reset cross-attn cache per chunk (upstream does this every chunk).
        for blk in crossattn_cache:
            blk["is_init"] = False

        latents = noise_fcHW[:, f0:f0 + CHUNK_FRAMES]  # [B, 8, C, h, w]
        sched = FlowUniPCMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=1, use_dynamic_shifting=False
        )
        sched.set_timesteps(SAMPLING_STEPS, device=device, shift=TIMESTEP_SHIFT)
        for step, t in enumerate(sched.timesteps):
            timestep = t * torch.ones([1, CHUNK_FRAMES], device=device, dtype=torch.float32)
            cap["on"] = chunk == 0 and step == 0
            with torch.no_grad():
                flow_pred = run_model(latents, timestep, current_start)
            cap["on"] = False
            # Raw velocity in CTHW [C, 8, h, w] (isolates the DiT forward).
            _dump(flow_pred[0].permute(1, 0, 2, 3).contiguous(),
                  args.out / f"py_c{chunk}_s{step}_vel.bin")
            latents = sched.step(flow_pred, t, latents, return_dict=False)[0]

        output[:, f0:f0 + CHUNK_FRAMES] = latents
        # CTHW [C, 8, h, w] for the engine byte-compare.
        chunk_cthw = latents[0].permute(1, 0, 2, 3).contiguous()
        _dump(chunk_cthw, args.out / f"py_chunk{chunk}_post.bin")
        _summarize(f"chunk{chunk}_post", chunk_cthw)

        # clean-context recache pass (timestep 0): commits this chunk's K/V.
        with torch.no_grad():
            run_model(latents, torch.zeros([1, CHUNK_FRAMES], device=device, dtype=torch.float32),
                      current_start)
    print(f"AR loop done in {time.time() - t0:.1f}s", flush=True)

    # pre-VAE latent in CTHW [C, f_lat, h, w] (== engine denoise_ar return).
    pre_vae_cthw = output[0].permute(1, 0, 2, 3).contiguous()  # [C, f_lat, h, w]
    _dump(pre_vae_cthw, args.out / "py_pre_vae_latent.bin")
    _summarize("pre_vae_latent (CTHW, raw)", pre_vae_cthw)

    del model, kv_cache, crossattn_cache
    gc.collect()

    # --- 3. VAE decode (fp32), same as the FastWan ref ---
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        _base_repo_dir(), subfolder="vae", torch_dtype=torch.float32
    ).eval()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, Z_DIM, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32).view(1, Z_DIM, 1, 1, 1)
    z = pre_vae_cthw.float().unsqueeze(0) * std + mean  # [1, C, f_lat, h, w]
    with torch.no_grad():
        video = vae.decode(z, return_dict=False)[0]  # [1, 3, frames, H, W]
    rgb = video[0]
    _summarize("vae_rgb (CTHW [-1, 1])", rgb)
    _dump(rgb, args.out / "py_vae_rgb.bin")

    print("[py-longlive-e2e] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
