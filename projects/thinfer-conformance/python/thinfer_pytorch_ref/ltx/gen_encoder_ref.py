"""LTX-2.3 text-conditioning reference (Gemma-3-12B encoder + FeatureExtractor V2).

Loads Gemma-3-12B FROM THE SAME QAT GGUF the engine loads, streaming it
LAYER-BY-LAYER (one reused `Gemma3DecoderLayer`, row-gathered token_embd) so peak
private RAM stays ~3.8GB (P0-proven). Produces ALL 49 hidden states, then runs
FeatureExtractor V2 (per-token RMS over D, flatten d*49+l, sqrt rescale, the two
`*_aggregate_embed` Linears from the connector safetensors) to the per-modality
caption projection (video [n,4096], audio [n,2048]).

Real-tokens-only: LTX left-pads to 1024 with mask-cumsum positions, so the n real
tokens at positions 0..n-1 under a plain causal mask are BIT-IDENTICAL to the
padded path (pad rows are zeroed by FE V2 + register-replaced by the connector).
So we encode just the prompt tokens -- the engine does the same. The 8-layer gated
connector (S=1024 framing with learnable registers) is a separate ref stage.

Dumps (LE):
  token_ids.bin   u32 [n]
  all_hidden.bin  f32 [49, n, 3840]   (embedding layer + 48 decoder layers)
  video_embed.bin f32 [n, 4096]       (FE V2 video aggregate-embed)
  audio_embed.bin f32 [n, 2048]       (FE V2 audio aggregate-embed)
  meta.txt        "n n_states hidden v_dim a_dim"

  uv run --with gguf --with tokenizers python -m thinfer_pytorch_ref.ltx.gen_encoder_ref \\
      --gguf <gemma...gguf> --connector <...embeddings_connectors.safetensors> \\
      --tokenizer <tokenizer.json> --prompt "..." --out <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Gemma-3-12B hyperparams (GGUF gemma3.* KV). Mirrors `ltx::gemma` in Rust.
VOCAB, HIDDEN, FFN, N_LAYERS = 262_208, 3840, 15360, 48
N_HEADS, N_KV, HEAD_DIM, EPS = 16, 8, 256, 1e-6
SLIDING_WINDOW, SLIDING_PATTERN, QUERY_PRE_ATTN = 1024, 6, 256
GLOBAL_THETA, LOCAL_THETA, ROPE_LINEAR_FACTOR = 1_000_000.0, 10_000.0, 8.0
N_STATES = N_LAYERS + 1  # 49
FE_EPS = 1e-6


def _dequant(t) -> np.ndarray:
    import gguf

    deq = getattr(gguf, "dequantize", None)
    out = deq(t.data, t.tensor_type) if deq else gguf.quants.dequantize(t.data, t.tensor_type)
    return np.ascontiguousarray(out).astype(np.float32).reshape(-1)


GGUF2HF = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
}


def gemma_all_hidden(gguf_path: Path, ids: list[int]) -> torch.Tensor:
    """Stream the GGUF layer-by-layer; return the 49 hidden states [49, n, D] bf16.
    Real tokens only: positions 0..n-1, plain causal mask (== LTX's left-pad path)."""
    import gguf
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3DecoderLayer,
        Gemma3RotaryEmbedding,
    )

    torch.set_grad_enabled(False)
    cfg = Gemma3TextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=FFN,
        num_hidden_layers=N_LAYERS, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV, head_dim=HEAD_DIM, rms_norm_eps=EPS,
        sliding_window=SLIDING_WINDOW, sliding_window_pattern=SLIDING_PATTERN,
        query_pre_attn_scalar=QUERY_PRE_ATTN, rope_theta=GLOBAL_THETA,
        rope_local_base_freq=LOCAL_THETA,
        rope_scaling={"rope_type": "linear", "factor": ROPE_LINEAR_FACTOR},
        attention_bias=False, hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
    )
    cfg._attn_implementation = "eager"
    n = len(ids)
    reader = gguf.GGUFReader(str(gguf_path))
    tmap = {t.name: t for t in reader.tensors}

    # Match the engine's precision so the band is the algorithm, not dtype noise:
    # the engine runs F32 acts (head_dim 256 needs large-D SDPA, which has no bf16
    # path; the residual also overflows f16). Every per-layer weight the engine
    # touches is bf16 on the GPU: matmul weights dequant Q8_0 -> bf16, and the F32
    # GGUF norm gains upload as bf16 too (the rmsnorm op reads bf16 weights). Only
    # the embedding gather stays f32. So here: f32 compute, ALL layer weights
    # (matmul + norm) bf16-rounded, embed f32.
    _f32_ref = os.environ.get("LTX_REF_F32")  # diag: full-f32 weights

    def _w(hf: str, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).float()
        return t if _f32_ref else t.to(torch.bfloat16).float()

    # embed: row-gather only the prompt tokens (Q8_0 rows block-aligned), x sqrt(D).
    te = tmap["token_embd.weight"]
    rows = torch.empty(n, HIDDEN, dtype=torch.float32)
    for j, tid in enumerate(ids):
        rows[j] = torch.from_numpy(gguf.dequantize(te.data[tid], te.tensor_type)).float()
    hidden = (rows * (HIDDEN**0.5)).unsqueeze(0)  # [1,n,D] f32

    position_ids = torch.arange(n).unsqueeze(0)
    rotary = Gemma3RotaryEmbedding(cfg)
    pos_emb = {
        "full_attention": rotary(hidden, position_ids, "full_attention"),
        "sliding_attention": rotary(hidden, position_ids, "sliding_attention"),
    }
    neg = torch.finfo(torch.float32).min
    q = torch.arange(n).view(n, 1)
    k = torch.arange(n).view(1, n)
    causal = k <= q
    mask = torch.where(causal, 0.0, neg).view(1, 1, n, n)  # n<<1024 -> sliding==full

    layer = Gemma3DecoderLayer(cfg, 0).eval()  # f32 compute
    params = dict(layer.named_parameters())
    states = [hidden.squeeze(0).clone()]
    for i in range(N_LAYERS):
        for gg, hf in GGUF2HF.items():
            p = params[hf]
            p.copy_(_w(hf, _dequant(tmap[f"blk.{i}.{gg}"])).reshape(p.shape))
        lt = cfg.layer_types[i]
        hidden = layer(hidden, position_embeddings=pos_emb[lt], attention_mask=mask,
                       position_ids=position_ids)
        states.append(hidden.squeeze(0).clone())
    return torch.stack(states, dim=0)  # [49, n, D] f32


def feature_extractor_v2(all_hidden: torch.Tensor, connector_path: Path):
    """FE V2: per-token RMS over D per-layer (eps 1e-6, no weight) -> flatten
    [n, D, L] C-order -> [n, D*L] (index d*L+l) -> rescale sqrt(out/D) -> the two
    aggregate-embed Linears. all_hidden: [L=49, n, D]."""
    from safetensors.torch import load_file

    L, n, D = all_hidden.shape
    enc = all_hidden.permute(1, 2, 0).float()  # [n, D, L]
    var = enc.pow(2).mean(dim=1, keepdim=True)  # over D
    normed = enc * torch.rsqrt(var + FE_EPS)  # [n, D, L]
    flat = normed.reshape(n, D * L)  # C-order: index = d*L + l

    w = load_file(str(connector_path))
    vW = w["text_embedding_projection.video_aggregate_embed.weight"].float()
    vB = w["text_embedding_projection.video_aggregate_embed.bias"].float()
    aW = w["text_embedding_projection.audio_aggregate_embed.weight"].float()
    aB = w["text_embedding_projection.audio_aggregate_embed.bias"].float()
    v_dim, a_dim = vW.shape[0], aW.shape[0]
    video = torch.nn.functional.linear(flat * (v_dim / D) ** 0.5, vW, vB)  # [n,4096]
    audio = torch.nn.functional.linear(flat * (a_dim / D) ** 0.5, aW, aB)  # [n,2048]
    return video, audio


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 then back to f32 (engine uploads bf16 weights, F32 acts)."""
    return x.to(torch.bfloat16).float()


def run_connector(
    dit_gguf: Path,
    prefix: str,
    feat: torch.Tensor,
    inner_dim: int,
    n_heads: int,
    head_dim: int,
    n_layers: int = 8,
    seq_len: int = 1024,
):
    """Run one modality's 8-layer gated connector (upstream `Embeddings1DConnector`)
    on the FE V2 aggregate `feat` `[n_real, inner]`. Mirrors engine precision: f32
    compute, bf16-rounded weights AND registers (the engine reads the registers
    back from their bf16 GPU upload). Frames `feat` valid-front into `[1, S, inner]`;
    the connector replaces the pad slots with `learnable_registers[s % 128]` and
    runs full bidirectional attention. Returns `[S, inner]` f32."""
    import sys

    here = Path(__file__).resolve()
    ltx_src = next(
        (
            p / "third-party" / "LTX-2" / "packages" / "ltx-core" / "src"
            for p in here.parents
            if (p / "third-party" / "LTX-2").is_dir()
        ),
        None,
    )
    assert ltx_src is not None, "could not locate third-party/LTX-2 above gen_encoder_ref.py"
    sys.path.insert(0, str(ltx_src))
    import gguf
    from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector

    torch.set_grad_enabled(False)
    reader = gguf.GGUFReader(str(dit_gguf))
    tmap = {t.name: t for t in reader.tensors}

    conn = Embeddings1DConnector(
        attention_head_dim=head_dim,
        num_attention_heads=n_heads,
        num_layers=n_layers,
        positional_embedding_theta=10000.0,
        positional_embedding_max_pos=[4096],
        num_learnable_registers=128,
        double_precision_rope=True,
        apply_gated_attention=True,
    ).float()

    state = dict(conn.named_parameters())
    for name, p in state.items():
        t = tmap[f"{prefix}.{name}"]
        arr = _dequant(t)
        p.copy_(_bf16(torch.from_numpy(arr).float().reshape(p.shape)))

    n = feat.shape[0]
    h = torch.zeros(1, seq_len, inner_dim, dtype=torch.float32)
    h[0, :n] = feat
    add_mask = torch.zeros(1, 1, 1, seq_len, dtype=torch.float32)
    add_mask[..., n:] = torch.finfo(torch.float32).min
    out, _ = conn(h, additive_attention_mask=add_mask)
    return out[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--connector", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path, help="product tokenizer.json (role TOKENIZER)")
    p.add_argument("--dit-gguf", type=Path, default=None, help="DiT GGUF for connector blocks")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Tokenize with the PRODUCT tokenizer.json via the same HF `tokenizers` lib the
    # Rust engine wraps (engine: `HfTokenizer::encode(prompt.trim(), true)`). NOT
    # `AutoTokenizer(gguf_file=...)`, whose GGUF-reconstructed tokenizer is degenerate
    # (splits words into char-fragments, ~2x the token count) and is NOT what serve
    # runs -- feeding the engine those ids validated correct math on the wrong
    # token distribution.
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))
    ids = tok.encode(args.prompt.strip(), add_special_tokens=True).ids
    print(f"[ltx-enc] prompt -> {len(ids)} tokens: {ids[:12]}...", flush=True)
    np.asarray(ids, dtype="<u4").tofile(str(args.out / "token_ids.bin"))

    all_hidden = gemma_all_hidden(args.gguf, ids)  # [49, n, D] bf16
    video, audio = feature_extractor_v2(all_hidden, args.connector)

    all_hidden.float().numpy().astype("<f4").tofile(str(args.out / "all_hidden.bin"))
    video.numpy().astype("<f4").tofile(str(args.out / "video_embed.bin"))
    audio.numpy().astype("<f4").tofile(str(args.out / "audio_embed.bin"))
    (args.out / "meta.txt").write_text(
        f"{len(ids)} {all_hidden.shape[0]} {HIDDEN} {video.shape[1]} {audio.shape[1]}\n"
    )
    print(
        f"[ltx-enc] all_hidden{tuple(all_hidden.shape)} "
        f"video{tuple(video.shape)} range[{video.min():.3f},{video.max():.3f}] "
        f"audio{tuple(audio.shape)} range[{audio.min():.3f},{audio.max():.3f}]",
        flush=True,
    )

    if args.dit_gguf is not None:
        import gc

        v_dim, a_dim = video.shape[1], audio.shape[1]
        vc = run_connector(args.dit_gguf, "video_embeddings_connector", video, v_dim, 32, 128)
        vc.numpy().astype("<f4").tofile(str(args.out / "video_connected.bin"))
        print(f"[ltx-enc] video_connected{tuple(vc.shape)} range[{vc.min():.3f},{vc.max():.3f}]", flush=True)
        del vc
        gc.collect()
        ac = run_connector(args.dit_gguf, "audio_embeddings_connector", audio, a_dim, 32, 64)
        ac.numpy().astype("<f4").tofile(str(args.out / "audio_connected.bin"))
        print(f"[ltx-enc] audio_connected{tuple(ac.shape)} range[{ac.min():.3f},{ac.max():.3f}]", flush=True)
        (args.out / "conn_meta.txt").write_text(f"1024 {v_dim} {a_dim}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
