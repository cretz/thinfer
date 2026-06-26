"""LTX-2.3 dual-stream DiT one-block reference (`BasicAVTransformerBlock`).

Instantiates the REAL upstream `LTXModel` (`third-party/LTX-2/.../transformer/
model.py`) with `num_layers=1`, loads block-0 + top-level weights FROM THE SAME
DiT GGUF the engine loads (Q8_0 matmuls dequant -> bf16; F32 norms/biases/tables
-> bf16; matches the engine upload precision), then runs the preprocessors + the
single block and dumps everything the engine's `forward_block_dumped` consumes:
the post-patchify token streams, the cross-attn caption KV, the per-stream AdaLN
modulation vectors, the rope position bounds, and the block output.

This isolates the BLOCK + rope-application (the novel P3 core: 5 attn sublayers,
per-head gating, av-cross, split/half-rot rope) from the patchifier (positions
are made up here; the engine rebuilds identical freqs from the same dumped
bounds) and the e2e two-stage path (P4).

Dumps (LE f32 unless noted):
  vx_in.bin  [Tv, 4096]      ax_in.bin  [Ta, 2048]   (post patchify_proj)
  vtext.bin  [Tvt, 4096]     atext.bin  [Tat, 2048]  (cross-attn caption KV)
  vx_out.bin [Tv, 4096]      ax_out.bin [Ta, 2048]   (block output)
  v_pos.bin  [3, Tv, 2]      a_pos.bin  [1, Ta, 2]    (rope grid bounds)
  vmod_*.bin / amod_*.bin    [inner]                 (16 AdaLN vectors / stream)
  meta.txt   "Tv Ta Tvt Tat"

  uv run --with gguf python -m thinfer_pytorch_ref.ltx.gen_dit_ref \\
      --dit-gguf <...Q8_0.gguf> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Tiny dims (B=1). Exercises every sublayer; av-cross needs both streams present.
TV, TA, TVT, TAT = 4, 4, 6, 6
SIGMA = 0.7

VIDEO_MAX_POS = [20, 2048, 2048]
AUDIO_MAX_POS = [20]


def _ltx_src() -> Path:
    here = Path(__file__).resolve()
    src = next(
        (
            p / "third-party" / "LTX-2" / "packages" / "ltx-core" / "src"
            for p in here.parents
            if (p / "third-party" / "LTX-2").is_dir()
        ),
        None,
    )
    assert src is not None, "could not locate third-party/LTX-2 above gen_dit_ref.py"
    return src


def _dequant(t) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    out = deq(t.data, t.tensor_type) if deq else gguf.quants.dequantize(t.data, t.tensor_type)
    return np.ascontiguousarray(out).astype(np.float32)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 then back to f32 (matches the engine weight upload)."""
    return x.to(torch.bfloat16).float()


def build_model(num_layers: int = 1):
    sys.path.insert(0, str(_ltx_src()))
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.rope import LTXRopeType

    torch.set_grad_enabled(False)
    model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        num_layers=num_layers,
        cross_attention_dim=4096,
        norm_eps=1e-6,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=VIDEO_MAX_POS,
        timestep_scale_multiplier=1000,
        use_middle_indices_grid=True,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        audio_positional_embedding_max_pos=AUDIO_MAX_POS,
        av_ca_timestep_scale_multiplier=1000,
        rope_type=LTXRopeType.SPLIT,
        double_precision_rope=True,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    ).eval()
    return model


def load_weights(model, dit_gguf: Path):
    import gguf

    reader = gguf.GGUFReader(str(dit_gguf))
    tmap = {t.name: t for t in reader.tensors}
    missing = []
    for name, p in model.named_parameters():
        t = tmap.get(name)
        if t is None:
            missing.append(name)
            continue
        arr = _dequant(t).reshape(tuple(p.shape))
        p.copy_(_bf16(torch.from_numpy(arr)))
    assert not missing, f"missing {len(missing)} params in GGUF, e.g. {missing[:6]}"


def make_positions(n_dims: int, seq: int, seed: int) -> torch.Tensor:
    """Deterministic [1, n_dims, seq, 2] start/end grid bounds (both engine and
    pyref consume the same dump, so any in-range bounds exercise the rope)."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, 12, (n_dims, seq), generator=g).float()
    pos = torch.stack([starts, starts + 1.0], dim=-1)  # [n_dims, seq, 2]
    return pos.unsqueeze(0)


def squeeze_vec(t: torch.Tensor) -> np.ndarray:
    """[1, 1, inner] (or [1, T, inner] with uniform rows) -> [inner] f32."""
    return t.reshape(t.shape[0], -1, t.shape[-1])[0, 0].float().numpy().astype("<f4")


def stream_mod(block, args, table, prompt_table) -> dict[str, np.ndarray]:
    """Extract the 16 AdaLN modulation vectors for one stream (see dit.rs
    `HostStreamMod`). Order/semantics mirror `transformer.py` `get_ada_values`."""
    B = args.x.shape[0]
    shift_msa, scale_msa, gate_msa = block.get_ada_values(table, B, args.timesteps, slice(0, 3))
    shift_mlp, scale_mlp, gate_mlp = block.get_ada_values(table, B, args.timesteps, slice(3, 6))
    shift_q, scale_q, gate_q = block.get_ada_values(table, B, args.timesteps, slice(6, 9))
    shift_kv, scale_kv = (
        prompt_table[None, None]
        + args.prompt_timestep.reshape(B, args.prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    return {
        "msa_scale": squeeze_vec(scale_msa),
        "msa_shift": squeeze_vec(shift_msa),
        "msa_gate": squeeze_vec(gate_msa),
        "cq_scale": squeeze_vec(scale_q),
        "cq_shift": squeeze_vec(shift_q),
        "cq_gate": squeeze_vec(gate_q),
        "ckv_scale": squeeze_vec(scale_kv),
        "ckv_shift": squeeze_vec(shift_kv),
        "mlp_scale": squeeze_vec(scale_mlp),
        "mlp_shift": squeeze_vec(shift_mlp),
        "mlp_gate": squeeze_vec(gate_mlp),
    }


def av_mod(block, args, av_table) -> dict[str, np.ndarray]:
    """The 5 av-cross vectors for one stream: a2v/v2a scale+shift (slices 0:2 /
    2:4) and the row-4 gate (a2v gate for video table, v2a gate for audio)."""
    B = args.x.shape[0]
    a2v_scale, a2v_shift, gate = block.get_av_ca_ada_values(
        av_table, B, args.cross_scale_shift_timestep, args.cross_gate_timestep, slice(0, 2)
    )
    v2a_scale, v2a_shift, _ = block.get_av_ca_ada_values(
        av_table, B, args.cross_scale_shift_timestep, args.cross_gate_timestep, slice(2, 4)
    )
    return {
        "a2v_scale": squeeze_vec(a2v_scale),
        "a2v_shift": squeeze_vec(a2v_shift),
        "v2a_scale": squeeze_vec(v2a_scale),
        "v2a_shift": squeeze_vec(v2a_shift),
        "av_gate": squeeze_vec(gate),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit-gguf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out

    model = build_model(num_layers=1)
    load_weights(model, args.dit_gguf)
    block = model.transformer_blocks[0]

    from ltx_core.model.transformer.modality import Modality

    gen = torch.Generator().manual_seed(1234)
    v_latent = torch.randn(1, TV, 128, generator=gen)
    a_latent = torch.randn(1, TA, 128, generator=gen)
    v_ctx = torch.randn(1, TVT, 4096, generator=gen)
    a_ctx = torch.randn(1, TAT, 2048, generator=gen)
    v_pos = make_positions(3, TV, 11)
    a_pos = make_positions(1, TA, 22)
    sigma = torch.tensor([SIGMA])
    v_ts = torch.full((1, 1), SIGMA)
    a_ts = torch.full((1, 1), SIGMA)

    video = Modality(latent=v_latent, sigma=sigma, timesteps=v_ts, positions=v_pos, context=v_ctx)
    audio = Modality(latent=a_latent, sigma=sigma, timesteps=a_ts, positions=a_pos, context=a_ctx)

    v_args = model.video_args_preprocessor.prepare(video, audio)
    a_args = model.audio_args_preprocessor.prepare(audio, video)

    # block output (num_layers=1; handles the perturbation-mask plumbing).
    v_out, a_out = model._process_transformer_blocks(v_args, a_args, None)

    # full DiT forward (patchify -> block -> output stage) -> velocity predictions.
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig

    v_vel, a_vel = model(video, audio, BatchedPerturbationConfig.empty(1))

    # modulation vectors (computed from the prepared args; the block-input
    # processor only attaches perturbation flags, leaving timesteps untouched).
    vmod = stream_mod(block, v_args, block.scale_shift_table, block.prompt_scale_shift_table)
    vmod.update(av_mod(block, v_args, block.scale_shift_table_a2v_ca_video))
    amod = stream_mod(
        block, a_args, block.audio_scale_shift_table, block.audio_prompt_scale_shift_table
    )
    amod.update(av_mod(block, a_args, block.scale_shift_table_a2v_ca_audio))

    def dump(name: str, t):
        arr = t.detach().float().numpy() if isinstance(t, torch.Tensor) else t
        np.ascontiguousarray(arr).astype("<f4").tofile(str(out / f"{name}.bin"))

    dump("vx_in", v_args.x[0])
    dump("ax_in", a_args.x[0])
    dump("vtext", v_args.context[0])
    dump("atext", a_args.context[0])
    dump("vx_out", v_out.x[0])
    dump("ax_out", a_out.x[0])
    dump("v_pos", v_pos[0])
    dump("a_pos", a_pos[0])
    # full-forward inputs/outputs (raw latents + velocity).
    dump("v_latent", v_latent[0])
    dump("a_latent", a_latent[0])
    dump("v_vel", v_vel[0])
    dump("a_vel", a_vel[0])
    (out / "sigma.txt").write_text(f"{SIGMA}\n")
    for k, v in vmod.items():
        dump(f"vmod_{k}", v)
    for k, v in amod.items():
        dump(f"amod_{k}", v)
    (out / "meta.txt").write_text(f"{TV} {TA} {TVT} {TAT}\n")

    print(
        f"[ltx-dit] vx_out{tuple(v_out.x.shape)} range[{v_out.x.min():.3f},{v_out.x.max():.3f}] "
        f"ax_out{tuple(a_out.x.shape)} range[{a_out.x.min():.3f},{a_out.x.max():.3f}]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
