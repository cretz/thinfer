"""MiniMax-H3 truncated DiT reference: token refiner + the first N blocks + final layer.

A torch transcription of the upstream DiT (`third-party/minimax-h3-mlx/reference/diffusers/
transformer_minimax_h3.py`, cross-checked against `minimax_h3_mlx/dit.py`) run over the SAME
weights the engine loads: every tensor is dequantized straight from the DiT GGUF to f32 (the
truth for those weights; the engine rounds to bf16 on upload). Weights are pulled one block at
a time, so host RAM stays at one block (~3 GiB) however many blocks run.

The packed layout (positions, tags, row timesteps) comes from the reference's own
`minimax_h3_mlx.packing` / `scheduler` (numpy; `mlx` is stubbed), so the engine's packing and
schedule are checked against upstream too, not against themselves.

Attention runs in the DIFFUSERS layout (`reorder_interleaved_qkv` + contiguous split into
to_q/to_k/to_v, `ff.net.0.proj` = `[value; gate]`), which is the module tree the turbo LoRA was
trained on. With `--lora` the adapter is applied exactly as PEFT does on that tree:
`y += (alpha / rank) * (x @ A^T) @ B^T`, alpha 8 per the lightx2v inference script.

Dumps (LE f32 unless noted), S = packed rows, H = 5376:
  video_in [Nv, 96]  audio_in [Na, 32]  text_in [Nt, 5120]          (forward inputs)
  positions [S, 3]  tags [S] i32  ts_index [S] u32  timesteps [T]  (layout + schedule)
  adaln_block{i} [3T, 6H]  adaln_final [T, 2H]                     (modulation tables)
  x_embed [S, H]                                (input projections, pre-refiner)
  refiner{j} [Nt, H]  refiner_out [Nt, H]       (per refiner block, then final_norm)
  block{i} [S, H]                               (residual stream after block i)
  refiner0.<stage> / block0.<stage>             (intra-block, engine `probe_layout` order:
      attn_in qkv q_normed attn_out attn_proj after_attn ffn_up swiglu ffn_down)
  final_norm [S, H]                             (modulated final norm)
  video_vel [Nv, 96]  audio_vel [Na, 32]
  meta.txt "Nt frames lat_h lat_w audio_latents steps step n_blocks"

  uv run --with gguf python -m thinfer_pytorch_ref.minimax.gen_dit_ref \\
      --dit-gguf <...Q4_K_M.gguf> [--lora <turbo.safetensors>] --out <dir>
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# A few hundred packed rows: 48 text + 2 x 20 audio + 3 frames x (6 x 8) video = 232. Every
# modality, two distinct timesteps (step 1 of a 4-step run puts video and audio at different
# levels), and M large enough to reach the engine's tiled matmul paths.
TEXT_ROWS = 48
LATENT_FRAMES = 3
LATENT_H = 12
LATENT_W = 16
AUDIO_LATENTS = 20
STEPS = 4
STEP = 1

HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
TEXT_DIM = 5120
VIDEO_PATCH_DIM = 96
AUDIO_DIM = 32
FREQ_DIM = 256
TIME_HIDDEN = 5376
TIME_DIM = 2688
ROPE_N = 16
ROPE_THETA = 10000.0
EPS = 1e-5
MODALITY_NUM = 3
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
LORA_ALPHA = 8.0


def _mlx_ref_root() -> Path:
    here = Path(__file__).resolve()
    root = next(
        (p / "third-party" / "minimax-h3-mlx" for p in here.parents if (p / "third-party" / "minimax-h3-mlx").is_dir()),
        None,
    )
    assert root is not None, "could not locate third-party/minimax-h3-mlx above gen_dit_ref.py"
    return root


def _import_reference():
    """Import upstream `packing` / `scheduler` with `mlx` stubbed by numpy (layout code only)."""
    core = types.ModuleType("mlx.core")
    core.array = lambda x, dtype=None: np.asarray(x, dtype=dtype)
    core.float32 = np.float32
    core.int32 = np.int32
    core.Dtype = object
    nn = types.ModuleType("mlx.nn")
    nn.Module = object
    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx.nn = nn
    sys.modules.update({"mlx": mlx, "mlx.core": core, "mlx.nn": nn})
    sys.path.insert(0, str(_mlx_ref_root()))
    from minimax_h3_mlx import packing, scheduler

    return packing, scheduler


class Weights:
    """Lazy f32 view of the DiT GGUF (plus an optional PEFT LoRA)."""

    def __init__(self, gguf_path: Path, lora_path: Path | None):
        import gguf

        self._gguf = gguf
        self.reader = gguf.GGUFReader(str(gguf_path))
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.lora = None
        if lora_path is not None:
            from safetensors.torch import load_file

            self.lora = load_file(str(lora_path))

    def get(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        t = self.tensors[name]
        q = self._gguf
        out = q.quants.dequantize(t.data, t.tensor_type)
        arr = np.ascontiguousarray(out, dtype=np.float32)
        # The comfy converter reshapes rows that are not superblock multiples; element order
        # survives, so the true shape is a plain reshape.
        assert arr.size == int(np.prod(shape)), f"{name}: {arr.shape} vs {shape}"
        return torch.from_numpy(arr.reshape(shape))

    def lora_pair(self, prefix: str, site: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.lora is None:
            return None
        a = self.lora.get(f"{prefix}.{site}.lora_A.default.weight")
        b = self.lora.get(f"{prefix}.{site}.lora_B.default.weight")
        if a is None or b is None:
            return None
        return a.float(), b.float()


def linear(x, w, b=None, lora=None):
    y = x @ w.T
    if b is not None:
        y = y + b
    if lora is not None:
        a, bb = lora
        y = y + (LORA_ALPHA / a.shape[0]) * ((x @ a.T) @ bb.T)
    return y


def rms(x, w, eps=EPS):
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def rope_tables(positions: torch.Tensor):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 2 * ROPE_N, 2, dtype=torch.float32) / (2 * ROPE_N)))
    freqs = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin(), inv_freq


def apply_rope(x, cos, sin):
    # x: (seq, heads, head_dim); diffusers `_apply_rotary_emb`.
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    c, s = cos[:, None, :], sin[:, None, :]
    x1, x2 = xr.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return torch.cat((xr * c + rot * s, xp), dim=-1)


def block_forward(W: Weights, base: str, lora_base: str, x, rope, mod=None, idx=None, probe=None):
    """One block in the diffusers module layout. `mod` = 6 tables `(3T, H)` or None (refiner).

    `probe(name, tensor)` receives the intermediates the engine's `BlockProbe` captures, in the
    ENGINE's layouts (fused contiguous qkv, checkpoint `[gate; value]` fc1 output)."""
    tap = probe or (lambda _n, _t: None)
    # The GGUF's fused qkv_proj is ALREADY contiguous [q; k; v] (its converter undid the
    # original per-head interleave; verified against the official FL2VA tensor).
    qkv = W.get(f"{base}.attn.qkv_proj.weight", (3 * INNER, HIDDEN))
    wq, wk, wv = qkv.chunk(3, dim=0)
    del qkv
    qn = W.get(f"{base}.attn.q_norm.weight", (HEAD_DIM,))
    kn = W.get(f"{base}.attn.k_norm.weight", (HEAD_DIM,))
    wo = W.get(f"{base}.attn.out_proj.weight", (HIDDEN, INNER))
    n1 = W.get(f"{base}.norm1.weight", (HIDDEN,))
    n2 = W.get(f"{base}.norm2.weight", (HIDDEN,))

    def lp(site):
        return W.lora_pair(lora_base, site)

    def modulate(h, scale, shift):
        if mod is None:
            return h
        return h * (1.0 + mod[scale].index_select(0, idx)) + mod[shift].index_select(0, idx)

    def gate(y, g):
        return y if mod is None else mod[g].index_select(0, idx) * y

    h = modulate(rms(x, n1), 1, 0)
    tap("attn_in", h)
    q = linear(h, wq, lora=lp("attn.to_q")).unflatten(-1, (HEADS, HEAD_DIM))
    k = linear(h, wk, lora=lp("attn.to_k")).unflatten(-1, (HEADS, HEAD_DIM))
    v = linear(h, wv, lora=lp("attn.to_v")).unflatten(-1, (HEADS, HEAD_DIM))
    tap("qkv", torch.cat((q, k, v), dim=-2).reshape(len(h), -1))
    q, k = rms(q, qn), rms(k, kn)
    tap("q_normed", q)
    if rope is not None:
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    a = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
    )[0].transpose(0, 1)
    tap("attn_out", a)
    a = linear(a.reshape(-1, INNER), wo, lora=lp("attn.to_out.0"))
    tap("attn_proj", a)
    x = x + gate(a, 2)
    tap("after_attn", x)
    del wq, wk, wv, wo

    fc1 = W.get(f"{base}.mlp.fc1.weight", (2 * FFN, HIDDEN))
    # Checkpoint `[gate; value]` -> diffusers `ff.net.0.proj` `[value; gate]`.
    w_ff0 = torch.cat((fc1[FFN:], fc1[:FFN]), dim=0)
    del fc1
    fc2 = W.get(f"{base}.mlp.fc2.weight", (HIDDEN, FFN))
    h = modulate(rms(x, n2), 4, 3)
    value, g = linear(h, w_ff0, lora=lp("ff.net.0.proj")).chunk(2, dim=-1)
    tap("ffn_up", torch.cat((g, value), dim=-1))
    s = value * F.silu(g)
    tap("swiglu", s)
    y = linear(s, fc2, lora=lp("ff.net.2"))
    tap("ffn_down", y)
    return x + gate(y, 5)


def timestep_embedding(t: torch.Tensor) -> torch.Tensor:
    half = FREQ_DIM // 2
    exponent = -np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    emb = t.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


def dump(out: Path, name: str, t) -> None:
    arr = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if arr.dtype.kind == "f":
        arr = arr.astype("<f4")
    arr.tofile(out / f"{name}.bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-gguf", type=Path, required=True)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--step", type=int, default=STEP, help="denoise step the forward sits at")
    ap.add_argument(
        "--text",
        type=Path,
        default=None,
        help="real conditioning ([rows, 5120] f32, e.g. the encoder gate's hidden.bin) instead of randn",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    packing, scheduler = _import_reference()
    W = Weights(args.dit_gguf, args.lora)
    real_text = None
    if args.text is not None:
        real_text = torch.from_numpy(np.fromfile(args.text, dtype="<f4").reshape(-1, TEXT_DIM))
    text_rows = TEXT_ROWS if real_text is None else real_text.shape[0]
    step = args.step

    # --- layout + schedule, from upstream ---
    layout = packing.build_packed_sequence(
        [1] * text_rows, LATENT_FRAMES, LATENT_H, LATENT_W, AUDIO_LATENTS, (1, 2, 2)
    )
    vs, au = scheduler.MiniMaxH3Scheduler(SHIFT_VIDEO), scheduler.MiniMaxH3Scheduler(SHIFT_AUDIO)
    # The turbo script: `steps` are model evaluations, the grid adds the terminal sigma.
    vs.set_timesteps(STEPS + 1)
    au.set_timesteps(STEPS + 1)
    per_step = [
        packing.build_row_timesteps(layout, float(tv), float(ta), float(tv), float(ta))
        for tv, ta in zip(vs.timesteps.tolist(), au.timesteps.tolist())
    ]
    timesteps = np.unique(np.concatenate([np.asarray(d) for d, _ in per_step])).astype(np.float32)
    distinct, inverse = per_step[step]
    remap = np.searchsorted(timesteps, np.asarray(distinct))
    ts_index = remap[np.asarray(inverse)].astype(np.int64)

    positions = torch.from_numpy(np.asarray(layout.position_ids, dtype=np.float32))
    tags = np.asarray(layout.token_tags, dtype=np.int64)
    vid_idx = torch.from_numpy(np.asarray(layout.video_indices, dtype=np.int64))
    aud_idx = torch.from_numpy(np.asarray(layout.audio_indices, dtype=np.int64))
    txt_idx = torch.from_numpy(np.asarray(layout.text_indices, dtype=np.int64))
    seq = layout.sequence_length
    adaln_idx = torch.from_numpy(ts_index * MODALITY_NUM + np.maximum(tags, 0))
    ts_idx_t = torch.from_numpy(ts_index)

    dump(out, "positions", positions)
    np.asarray(tags, dtype="<i4").tofile(out / "tags.bin")
    np.asarray(ts_index, dtype="<u4").tofile(out / "ts_index.bin")
    dump(out, "timesteps", timesteps)

    # --- inputs ---
    g = torch.Generator().manual_seed(1234)
    video_in = torch.randn(len(vid_idx), VIDEO_PATCH_DIM, generator=g)
    audio_in = torch.randn(len(aud_idx), AUDIO_DIM, generator=g)
    text_in = torch.randn(TEXT_ROWS, TEXT_DIM, generator=g) if real_text is None else real_text
    dump(out, "video_in", video_in)
    dump(out, "audio_in", audio_in)
    dump(out, "text_in", text_in)

    cos, sin, inv_freq = rope_tables(positions)
    ck = W.tensors.get("rope.inv_freq")
    if ck is not None:
        shipped = torch.from_numpy(np.asarray(ck.data, dtype=np.float32).reshape(-1))
        assert torch.allclose(shipped, inv_freq, rtol=1e-6), "rope.inv_freq disagrees with the formula"

    # --- input projections ---
    video_emb = linear(
        video_in,
        W.get("video_patch_proj.weight", (HIDDEN, VIDEO_PATCH_DIM)),
        W.get("video_patch_proj.bias", (HIDDEN,)),
    )
    audio_emb = linear(
        audio_in,
        W.get("audio_patch_proj.weight", (HIDDEN, AUDIO_DIM)),
        W.get("audio_patch_proj.bias", (HIDDEN,)),
    )
    text = linear(
        text_in,
        W.get("condition_proj.weight", (HIDDEN, TEXT_DIM)),
        W.get("condition_proj.bias", (HIDDEN,)),
    )
    x = torch.zeros(seq, HIDDEN)
    x[txt_idx] = text
    x[vid_idx] = video_emb
    x[aud_idx] = audio_emb
    dump(out, "x_embed", x)

    # --- token refiner ---
    def prober(prefix):
        return lambda name, t: dump(out, f"{prefix}.{name}", t)

    for j in range(2):
        text = block_forward(
            W,
            f"token_refiner.blocks.{j}",
            f"token_refiner.refiner_blocks.{j}",
            text,
            None,
            probe=prober(f"refiner{j}") if j == 0 else None,
        )
        dump(out, f"refiner{j}", text)
    text = rms(text, W.get("token_refiner.final_norm.weight", (HIDDEN,)))
    dump(out, "refiner_out", text)
    x[txt_idx] = text

    # --- timestep embedding (f32 throughout) ---
    temb = timestep_embedding(torch.from_numpy(timesteps))
    temb = linear(
        temb,
        W.get("time_embedder.proj_in.weight", (TIME_HIDDEN, FREQ_DIM)),
        W.get("time_embedder.proj_in.bias", (TIME_HIDDEN,)),
    )
    temb = linear(
        F.silu(temb),
        W.get("time_embedder.proj_out.weight", (TIME_DIM, TIME_HIDDEN)),
        W.get("time_embedder.proj_out.bias", (TIME_DIM,)),
    )
    a_temb = F.silu(temb)

    # --- blocks ---
    for i in range(args.blocks):
        proj = linear(
            a_temb,
            W.get(f"blocks.{i}.adaln_proj.linear.weight", (6 * MODALITY_NUM * HIDDEN, TIME_DIM)),
            W.get(f"blocks.{i}.adaln_proj.linear.bias", (6 * MODALITY_NUM * HIDDEN,)),
        ).view(-1, 6 * HIDDEN)
        dump(out, f"adaln_block{i}", proj)
        mod = proj.chunk(6, dim=-1)
        x = block_forward(
            W,
            f"blocks.{i}",
            f"transformer_blocks.{i}",
            x,
            (cos, sin),
            mod,
            adaln_idx,
            probe=prober(f"block{i}") if i == 0 else None,
        )
        dump(out, f"block{i}", x)
        print(f"block {i}: rms {x.pow(2).mean().sqrt():.4f}", flush=True)

    # --- final layer ---
    fin = linear(
        a_temb,
        W.get("final_layer.adaln_proj.linear.weight", (2 * HIDDEN, TIME_DIM)),
        W.get("final_layer.adaln_proj.linear.bias", (2 * HIDDEN,)),
    )
    dump(out, "adaln_final", fin)
    shift, scale = fin.chunk(2, dim=-1)
    h = rms(x, W.get("final_layer.norm.weight", (HIDDEN,)))
    h = h * (1.0 + scale.index_select(0, ts_idx_t)) + shift.index_select(0, ts_idx_t)
    dump(out, "final_norm", h)
    vel_v = linear(
        h,
        W.get("final_layer.video_out.weight", (VIDEO_PATCH_DIM, HIDDEN)),
        W.get("final_layer.video_out.bias", (VIDEO_PATCH_DIM,)),
    )[vid_idx]
    vel_a = linear(
        h,
        W.get("final_layer.audio_out.weight", (AUDIO_DIM, HIDDEN)),
        W.get("final_layer.audio_out.bias", (AUDIO_DIM,)),
    )[aud_idx]
    dump(out, "video_vel", vel_v)
    dump(out, "audio_vel", vel_a)

    (out / "meta.txt").write_text(
        f"{text_rows} {LATENT_FRAMES} {LATENT_H} {LATENT_W} {AUDIO_LATENTS} {STEPS} {step} {args.blocks}\n"
    )
    print(
        f"minimax dit ref: {seq} rows, {len(timesteps)} timesteps, {args.blocks} blocks, "
        f"lora={'on' if args.lora else 'off'}, video vel rms {vel_v.pow(2).mean().sqrt():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
